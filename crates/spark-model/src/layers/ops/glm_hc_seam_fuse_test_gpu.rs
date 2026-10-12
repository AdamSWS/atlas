// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): `ATLAS_GLM_HC_SEAM_FUSE`'s one launch against the launches
//! it replaces at GLM's shape (HC4, hidden 4096, BF16 highway), with and
//! without the finishing site's post, for every batch of 1..=32 rows:
//!
//! * serving's chain: `glm_hc_decode_{post_,}partial_ilp_bf16`,
//!   `glm_hc_decode_finalize_ilp_bf16`, `rms_norm_vanilla_regs`;
//! * the established chain: `glm_hc_decode_{post_,}partial_bf16` (FP32 mix
//!   weight, the BF16 one widened), `glm_hc_decode_finalize_bf16`,
//!   `rms_norm_vanilla`.
//!
//! Every byte of the highway, the collapsed rows, the next post and comb and
//! the normed rows is the same, and the counters settle on whole launches.
//! `fused_seam_rows_do_not_depend_on_the_batch` holds a probe row to
//! the same bytes at every width and slot. Launched with PDL (`ATLAS_PDL=1`)
//! as in serving.
//!
//!   cargo test --release -p spark-model --lib glm_hc_seam_fuse -- --ignored --nocapture

use super::*;
use spark_runtime::cuda_backend::AtlasCudaBackend;

const H: usize = 4096;
const K: usize = 4 * H;
const M: usize = 24;
const ITERS: u32 = 20;
const NORM_EPS: f32 = 1e-5;
const HC_EPS: f32 = 1e-6;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    // SAFETY: plain integer/float slices, read as their bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
        .to_vec()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn backend() -> Result<AtlasCudaBackend> {
    // Serving launches the seam with PDL; so do these tests (when the
    // process has not read the switch yet).
    // SAFETY: set before any thread of this test process reads it.
    unsafe { std::env::set_var("ATLAS_PDL", "1") };
    spark_runtime::cuda_backend::configure_pdl("glm-5.3-flash");
    AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
}

/// One site's weights: the BF16 mix weight and its FP32 widening, scale,
/// base, and an RMS norm weight.
struct Site {
    fn16: DevicePtr,
    fn32: DevicePtr,
    scale: DevicePtr,
    base: DevicePtr,
    norm_w: DevicePtr,
}

impl Site {
    fn new(gpu: &dyn GpuBackend, rng: &mut Rng) -> Result<Self> {
        let f: Vec<u16> = (0..M * K).map(|_| bf16(rng.unit() * 0.02)).collect();
        let f32w: Vec<f32> = f
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        let scale: Vec<f32> = vec![0.9 + 0.2 * rng.unit(), 0.8, 1.1];
        let base: Vec<f32> = (0..M).map(|_| rng.unit() * 0.5).collect();
        let w: Vec<u16> = (0..H).map(|_| bf16(1.0 + 0.2 * rng.unit())).collect();
        Ok(Self {
            fn16: upload(gpu, &bytes_of(&f))?,
            fn32: upload(gpu, &bytes_of(&f32w))?,
            scale: upload(gpu, &bytes_of(&scale))?,
            base: upload(gpu, &bytes_of(&base))?,
            norm_w: upload(gpu, &bytes_of(&w))?,
        })
    }
}

/// The inputs of a seam over `rows` rows: highway, finishing block output,
/// and the finishing site's post / comb.
#[derive(Clone)]
struct Rows {
    streams: Vec<u16>,
    block_out: Vec<u16>,
    post: Vec<f32>,
    comb: Vec<f32>,
}

impl Rows {
    fn random(rng: &mut Rng, rows: usize) -> Self {
        let mut comb = Vec::with_capacity(rows * 16);
        for _ in 0..rows * 4 {
            let r: Vec<f32> = (0..4).map(|_| 0.1 + (rng.unit() + 1.0) * 0.4).collect();
            let s: f32 = r.iter().sum();
            comb.extend(r.iter().map(|v| v / s));
        }
        Self {
            streams: (0..rows * K).map(|_| bf16(rng.unit() * 2.0)).collect(),
            block_out: (0..rows * H).map(|_| bf16(rng.unit() * 1.5)).collect(),
            post: (0..rows * 4)
                .map(|_| 2.0 / (1.0 + (-rng.unit() * 3.0).exp()))
                .collect(),
            comb,
        }
    }
    fn row(&self, t: usize) -> Self {
        Self {
            streams: self.streams[t * K..(t + 1) * K].to_vec(),
            block_out: self.block_out[t * H..(t + 1) * H].to_vec(),
            post: self.post[t * 4..(t + 1) * 4].to_vec(),
            comb: self.comb[t * 16..(t + 1) * 16].to_vec(),
        }
    }
    fn with_row(&self, t: usize, probe: &Self) -> Self {
        let mut out = self.clone();
        out.streams[t * K..(t + 1) * K].copy_from_slice(&probe.streams);
        out.block_out[t * H..(t + 1) * H].copy_from_slice(&probe.block_out);
        out.post[t * 4..(t + 1) * 4].copy_from_slice(&probe.post);
        out.comb[t * 16..(t + 1) * 16].copy_from_slice(&probe.comb);
        out
    }
}

/// Device buffers of one seam run; outputs start poisoned.
struct Bufs {
    rows: u32,
    streams: DevicePtr,
    block_out: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    partial: DevicePtr,
    hidden: DevicePtr,
    normed: DevicePtr,
}

/// Every output byte of a seam run.
#[derive(PartialEq)]
struct Out {
    streams: Vec<u8>,
    hidden: Vec<u8>,
    post: Vec<u8>,
    comb: Vec<u8>,
    normed: Vec<u8>,
}

impl Out {
    /// The first differing (buffer, byte) or None.
    fn first_diff(&self, other: &Self) -> Option<(&'static str, usize)> {
        [
            ("streams", &self.streams, &other.streams),
            ("hidden", &self.hidden, &other.hidden),
            ("post", &self.post, &other.post),
            ("comb", &self.comb, &other.comb),
            ("normed", &self.normed, &other.normed),
        ]
        .into_iter()
        .find_map(|(n, a, b)| {
            a.iter()
                .zip(b.iter())
                .position(|(x, y)| x != y)
                .map(|i| (n, i))
        })
    }
    /// Row `t` of every buffer.
    fn row(&self, t: usize) -> Vec<Vec<u8>> {
        vec![
            self.streams[t * K * 2..(t + 1) * K * 2].to_vec(),
            self.hidden[t * H * 2..(t + 1) * H * 2].to_vec(),
            self.post[t * 16..(t + 1) * 16].to_vec(),
            self.comb[t * 64..(t + 1) * 64].to_vec(),
            self.normed[t * H * 2..(t + 1) * H * 2].to_vec(),
        ]
    }
}

impl Bufs {
    fn new(gpu: &dyn GpuBackend, x: &Rows, poison: u8) -> Result<Self> {
        let rows = x.post.len() / 4;
        let p = |n: usize| upload(gpu, &vec![poison; n]);
        Ok(Self {
            rows: rows as u32,
            streams: upload(gpu, &bytes_of(&x.streams))?,
            block_out: upload(gpu, &bytes_of(&x.block_out))?,
            post: upload(gpu, &bytes_of(&x.post))?,
            comb: upload(gpu, &bytes_of(&x.comb))?,
            partial: p(64 * 32 * 25 * 4)?,
            hidden: p(rows * H * 2)?,
            normed: p(rows * H * 2)?,
        })
    }

    fn out(&self, gpu: &dyn GpuBackend) -> Result<Out> {
        gpu.synchronize(gpu.default_stream())?;
        let r = self.rows as usize;
        Ok(Out {
            streams: read(gpu, self.streams, r * K * 2)?,
            hidden: read(gpu, self.hidden, r * H * 2)?,
            post: read(gpu, self.post, r * 16)?,
            comb: read(gpu, self.comb, r * 64)?,
            normed: read(gpu, self.normed, r * H * 2)?,
        })
    }

    /// The three launches: partial (`ilp`: the BF16-weight ILP twins, else
    /// the established kernels on the FP32 weight), finalize, RMS norm.
    fn chain_on(
        &self,
        gpu: &dyn GpuBackend,
        site: &Site,
        post: bool,
        ilp: bool,
        s: u64,
    ) -> Result<()> {
        let m = "glm_hc_prefill_vec";
        let (partial, finalize, weight) = match (ilp, post) {
            (true, true) => (
                "glm_hc_decode_post_partial_ilp_bf16",
                "glm_hc_decode_finalize_ilp_bf16",
                site.fn16,
            ),
            (true, false) => (
                "glm_hc_decode_partial_ilp_bf16",
                "glm_hc_decode_finalize_ilp_bf16",
                site.fn16,
            ),
            (false, true) => (
                "glm_hc_decode_post_partial_bf16",
                "glm_hc_decode_finalize_bf16",
                site.fn32,
            ),
            (false, false) => (
                "glm_hc_decode_partial_bf16",
                "glm_hc_decode_finalize_bf16",
                site.fn32,
            ),
        };
        KernelLaunch::new(gpu, gpu.kernel(m, partial)?)
            .grid([64, self.rows.div_ceil(4), 1])
            .block([128, 1, 1])
            .arg_ptr(if post {
                self.block_out
            } else {
                DevicePtr::NULL
            })
            .arg_ptr(self.streams)
            .arg_ptr(self.post)
            .arg_ptr(self.comb)
            .arg_ptr(weight)
            .arg_ptr(self.partial)
            .arg_u32(self.rows)
            .launch(s)?;
        KernelLaunch::new(gpu, gpu.kernel(m, finalize)?)
            .grid([self.rows, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.streams)
            .arg_ptr(self.partial)
            .arg_ptr(site.scale)
            .arg_ptr(site.base)
            .arg_ptr(self.hidden)
            .arg_ptr(self.post)
            .arg_ptr(self.comb)
            .arg_u32(self.rows)
            .arg_u32(ITERS)
            .arg_f32(NORM_EPS)
            .arg_f32(HC_EPS)
            .launch(s)?;
        let norm = if ilp {
            gpu.kernel("glm_rms_norm_regs", "rms_norm_vanilla_regs")?
        } else {
            gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?
        };
        KernelLaunch::new(gpu, norm)
            .grid([self.rows, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(self.hidden)
            .arg_ptr(site.norm_w)
            .arg_ptr(self.normed)
            .arg_u32(H as u32)
            .arg_f32(NORM_EPS)
            .launch(s)
    }

    fn chain(&self, gpu: &dyn GpuBackend, site: &Site, post: bool, ilp: bool) -> Result<()> {
        self.chain_on(gpu, site, post, ilp, gpu.default_stream())
    }

    fn fused(&self, gpu: &dyn GpuBackend, site: &Site, post: bool) -> Result<()> {
        self.fused_on(gpu, site, post, gpu.default_stream())
    }

    /// The fused launch through its serving launcher.
    fn fused_on(&self, gpu: &dyn GpuBackend, site: &Site, post: bool, stream: u64) -> Result<()> {
        let launched = hc_seam_fused_for(
            gpu,
            true,
            &SeamFuse {
                block_out: if post {
                    self.block_out
                } else {
                    DevicePtr::NULL
                },
                streams: self.streams,
                post: self.post,
                comb: self.comb,
                hc_fn_bf16: site.fn16,
                hc_scale: site.scale,
                hc_base: site.base,
                partial: self.partial,
                hidden: self.hidden,
                norm_kernel: gpu.kernel("glm_rms_norm_regs", "rms_norm_vanilla_regs")?,
                norm_weight: site.norm_w,
                normed: self.normed,
                tokens: self.rows,
                sinkhorn_iters: ITERS,
                norm_eps: NORM_EPS,
                hc_eps: HC_EPS,
                rms_eps: NORM_EPS,
            },
            stream,
        )?;
        ensure!(launched, "the fused seam did not launch");
        Ok(())
    }
}

/// Every group counter at a whole number of launches (64 CTAs each).
fn counters_settled(gpu: &dyn GpuBackend) -> Result<bool> {
    let c = read(gpu, seam_counters(gpu)?, SEAM_COUNTER_BYTES)?;
    Ok(c.chunks(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .all(|v| v % 64 == 0))
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn fused_seam_matches_the_three_launches() -> Result<()> {
    let gpu = backend()?;
    eprintln!(
        "PDL launches: {}",
        spark_runtime::cuda_backend::pdl_enabled()
    );
    let mut rng = Rng(0x5eed_f00d_1234_5678);
    let site = Site::new(&gpu, &mut rng)?;
    for rows in 1..=32usize {
        let x = Rows::random(&mut rng, rows);
        for post in [false, true] {
            let fused = Bufs::new(&gpu, &x, 0xa5)?;
            fused.fused(&gpu, &site, post)?;
            let fused = fused.out(&gpu)?;
            ensure!(
                counters_settled(&gpu)?,
                "counters off a whole launch at {rows} rows"
            );
            for ilp in [true, false] {
                let chain = Bufs::new(&gpu, &x, 0x5a)?;
                chain.chain(&gpu, &site, post, ilp)?;
                let chain = chain.out(&gpu)?;
                if let Some((buf, at)) = fused.first_diff(&chain) {
                    bail!("rows {rows} post {post} ilp {ilp}: {buf} differs at byte {at}");
                }
            }
        }
        eprintln!("rows {rows}: fused == ILP chain == established chain (post and pre)");
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn fused_seam_rows_do_not_depend_on_the_batch() -> Result<()> {
    let gpu = backend()?;
    let mut rng = Rng(0x0dd_ba11_cafe_beef);
    let site = Site::new(&gpu, &mut rng)?;
    let probe = Rows::random(&mut rng, 1);
    for post in [false, true] {
        let alone = Bufs::new(&gpu, &probe, 0x11)?;
        alone.fused(&gpu, &site, post)?;
        let want = alone.out(&gpu)?.row(0);
        for rows in [2usize, 3, 4, 5, 8, 16, 31, 32] {
            let base = Rows::random(&mut rng, rows);
            for slot in [0, rows / 2, rows - 1] {
                let b = Bufs::new(&gpu, &base.with_row(slot, &probe.row(0)), 0x22)?;
                b.fused(&gpu, &site, post)?;
                ensure!(
                    b.out(&gpu)?.row(slot) == want,
                    "post {post}: the probe row differs at width {rows}, slot {slot}"
                );
            }
        }
    }
    ensure!(counters_settled(&gpu)?, "counters off a whole launch");
    Ok(())
}

/// Indicative only (03 serves other traffic): a CUDA graph of 50
/// back-to-back seams (PDL edges kept), the three launches against the
/// fused one, replayed; GPU time a seam, best of 15 alternating trials.
#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn fused_seam_timing() -> Result<()> {
    let gpu = backend()?;
    let mut rng = Rng(0x7177_ab1e);
    let site = Site::new(&gpu, &mut rng)?;
    let s = gpu.create_stream()?;
    seam_counters(&gpu)?;
    const SEAMS: u32 = 50;
    const REPLAYS: u32 = 20;
    for rows in [1usize, 2, 3, 4, 5, 6, 8, 16, 32] {
        let b = Bufs::new(&gpu, &Rows::random(&mut rng, rows), 0)?;
        for post in [true, false] {
            let mut graphs = Vec::new();
            for arm in 0..2 {
                gpu.begin_capture(s)?;
                for _ in 0..SEAMS {
                    if arm == 0 {
                        b.chain_on(&gpu, &site, post, true, s)?;
                    } else {
                        b.fused_on(&gpu, &site, post, s)?;
                    }
                }
                let graph = gpu.end_capture(s)?;
                gpu.launch_graph(graph, s)?;
                graphs.push(graph);
            }
            gpu.synchronize(s)?;
            // Trials alternate the arms, so drift in the other tenants' load
            // falls on both.
            let mut us = [f64::MAX; 2];
            for _ in 0..15 {
                for (arm, &graph) in graphs.iter().enumerate() {
                    let t0 = std::time::Instant::now();
                    for _ in 0..REPLAYS {
                        gpu.launch_graph(graph, s)?;
                    }
                    gpu.synchronize(s)?;
                    let per = t0.elapsed().as_secs_f64() * 1e6 / f64::from(SEAMS * REPLAYS);
                    us[arm] = us[arm].min(per);
                }
            }
            for graph in graphs {
                gpu.destroy_graph(graph)?;
            }
            eprintln!(
                "rows {rows:2} post {post:5}: three launches {:6.2} us, fused {:6.2} us a seam",
                us[0], us[1]
            );
        }
    }
    Ok(())
}
