//
// Created for the batch matrix multiplication of Cheon, Kang and Lee,
// "Fast Batch Matrix Multiplication in Ciphertexts".
//

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/BatchMatrix.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ConstantsGPU.cuh"
#include "CudaUtils.cuh"
#include "LimbUtils.cuh"
#include "ModMult.cuh"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace FIDESlib::CKKS {

namespace {

// ---------------------------------------------------------------------------
// Host modular arithmetic
// ---------------------------------------------------------------------------

uint64_t modpow(uint64_t base, uint64_t exp, uint64_t p) {
	uint64_t r = 1, x = base % p;
	while (exp) {
		if (exp & 1)
			r = static_cast<uint64_t>((static_cast<__uint128_t>(r) * x) % p);
		x = static_cast<uint64_t>((static_cast<__uint128_t>(x) * x) % p);
		exp >>= 1;
	}
	return r;
}

/** Valid for prime p. */
uint64_t modinv(uint64_t a, uint64_t p) {
	return modpow(a, p - 2, p);
}

/** Inverse modulo an arbitrary modulus; 2N is a power of two, so Fermat does not apply. */
uint64_t modinvGeneric(uint64_t a, uint64_t m) {
	int64_t t = 0, newt = 1;
	int64_t r = static_cast<int64_t>(m), newr = static_cast<int64_t>(a % m);
	while (newr != 0) {
		const int64_t q = r / newr;
		int64_t tmp		= t - q * newt;
		t = newt, newt = tmp;
		tmp = r - q * newr;
		r = newr, newr = tmp;
	}
	if (r != 1)
		throw std::runtime_error("value is not invertible modulo the cyclotomic order");
	if (t < 0)
		t += static_cast<int64_t>(m);
	return static_cast<uint64_t>(t);
}

/**
 * psi is a primitive 2k-th root of unity modulo p exactly when psi^k == -1, so
 * candidates can be tested directly without factoring p-1.
 *
 * Any primitive root works: the transform is inverted with the same root, so the
 * negacyclic convolution it computes does not depend on which one is chosen.
 */
uint64_t find2kthRoot(uint64_t p, int k) {
	const uint64_t order = 2ull * static_cast<uint64_t>(k);
	if ((p - 1) % order != 0)
		throw std::runtime_error("prime does not admit a 2k-th root of unity for the requested subring degree");
	const uint64_t e = (p - 1) / order;
	for (uint64_t a = 2; a < 4096; ++a) {
		const uint64_t psi = modpow(a, e, p);
		if (modpow(psi, k, p) == p - 1)
			return psi;
	}
	throw std::runtime_error("could not find a primitive 2k-th root of unity");
}

uint64_t shoupFactor(uint64_t b, uint64_t p) {
	return static_cast<uint64_t>((static_cast<__uint128_t>(b) << 64) / p);
}

uint32_t bitrev(uint32_t x, int bits) {
	uint32_t r = 0;
	for (int i = 0; i < bits; ++i) {
		r = (r << 1) | (x & 1u);
		x >>= 1;
	}
	return r;
}

int log2i(int x) {
	int r = 0;
	while ((1 << r) < x)
		++r;
	return r;
}

uint64_t centeredToModular(int64_t v, uint64_t p) {
	if (v >= 0)
		return static_cast<uint64_t>(v) % p;
	const uint64_t m = static_cast<uint64_t>(-v) % p;
	return m == 0 ? 0 : p - m;
}

// ---------------------------------------------------------------------------
// Twiddle tables for the length-k negacyclic transform
// ---------------------------------------------------------------------------

/**
 * Per-limb twiddle factors for the subring transform, in the bit-reversed
 * ordering used by the Cooley-Tukey / Gentleman-Sande butterflies.
 */
struct SubringTables {
	int k		 = 0;
	int d		 = 0;
	int numLimbs = 0;
	uint64_t* psi			 = nullptr; ///< [numLimbs][k] forward twiddles.
	uint64_t* psi_shoup		 = nullptr;
	uint64_t* psi_inv		 = nullptr; ///< [numLimbs][k] inverse twiddles.
	uint64_t* psi_inv_shoup	 = nullptr;
	uint64_t* kinv			 = nullptr; ///< [numLimbs] k^-1 mod p.
	uint64_t* kinv_shoup	 = nullptr;
	int* primeids			 = nullptr; ///< [numLimbs]

	// Tables for the partial transform between FIDESlib's length-N NTT domain
	// and the R_k NTT domain. omega = psi_N^(2k) is a primitive d-th root, and
	// each residue class of the length-N transform is a cyclic d-point DFT.
	uint64_t* wfwd			= nullptr; ///< [numLimbs][d] omega^m.
	uint64_t* wfwd_shoup	= nullptr;
	uint64_t* winv			= nullptr; ///< [numLimbs][d] omega^-m.
	uint64_t* winv_shoup	= nullptr;
	uint64_t* psiFwd		= nullptr; ///< [numLimbs][k][d] psi_N^(+h0(s)*i).
	uint64_t* psiInv		= nullptr; ///< [numLimbs][k][d] psi_N^(-h0(s)*i) / d.
};

std::mutex g_tablesMutex;
std::map<std::string, SubringTables> g_tables;

template <typename T> T* deviceCopy(const std::vector<T>& host) {
	T* dev = nullptr;
	cudaMalloc(&dev, host.size() * sizeof(T));
	cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
	return dev;
}

/**
 * Tables depend only on the subring degree and the set of RNS primes, so they
 * are built once per configuration and shared by every batch matrix operation.
 */
const SubringTables& getSubringTables(int k, const std::vector<int>& primeids, const std::vector<uint64_t>& primes, const std::vector<uint64_t>& psiN, int N, int device) {
	// The key must carry the prime values, not just their ids: two contexts can
	// use the same primeid slots for different primes (a different scaling
	// modulus or ring degree is enough), and the twiddles are derived from the
	// values. Keying on ids alone would silently hand back the wrong tables.
	std::string key = std::to_string(device) + ":" + std::to_string(k);
	for (size_t i = 0; i < primeids.size(); ++i)
		key += "," + std::to_string(primeids[i]) + "@" + std::to_string(primes[i]);

	std::lock_guard<std::mutex> lock(g_tablesMutex);
	auto it = g_tables.find(key);
	if (it != g_tables.end())
		return it->second;

	const int numLimbs = static_cast<int>(primeids.size());
	const int logk	   = log2i(k);
	const int d		   = N / k;

	std::vector<uint64_t> psi(static_cast<size_t>(numLimbs) * k), psi_s(static_cast<size_t>(numLimbs) * k);
	std::vector<uint64_t> psi_i(static_cast<size_t>(numLimbs) * k), psi_i_s(static_cast<size_t>(numLimbs) * k);
	std::vector<uint64_t> kinv(numLimbs), kinv_s(numLimbs);
	std::vector<uint64_t> wf(static_cast<size_t>(numLimbs) * d), wf_s(static_cast<size_t>(numLimbs) * d);
	std::vector<uint64_t> wi(static_cast<size_t>(numLimbs) * d), wi_s(static_cast<size_t>(numLimbs) * d);
	std::vector<uint64_t> pf(static_cast<size_t>(numLimbs) * k * d), pi(static_cast<size_t>(numLimbs) * k * d);

	for (int l = 0; l < numLimbs; ++l) {
		const uint64_t p = primes[l];

		// The subring root must be the one FIDESlib's own transform induces:
		// zeta = psi_N^d. Choosing an unrelated 2k-th root would still be a
		// valid transform on its own, but it would not line up with the
		// residue classes of the length-N NTT.
		const uint64_t root	 = modpow(psiN[l], static_cast<uint64_t>(d), p);
		if (modpow(root, k, p) != p - 1)
			throw std::runtime_error("psi_N^d is not a primitive 2k-th root of unity");
		const uint64_t iroot = modinv(root, p);
		for (int i = 0; i < k; ++i) {
			const uint32_t e			   = bitrev(static_cast<uint32_t>(i), logk);
			psi[static_cast<size_t>(l) * k + i]	  = modpow(root, e, p);
			psi_i[static_cast<size_t>(l) * k + i] = modpow(iroot, e, p);
			psi_s[static_cast<size_t>(l) * k + i]	= shoupFactor(psi[static_cast<size_t>(l) * k + i], p);
			psi_i_s[static_cast<size_t>(l) * k + i] = shoupFactor(psi_i[static_cast<size_t>(l) * k + i], p);
		}
		kinv[l]	  = modinv(static_cast<uint64_t>(k) % p, p);
		kinv_s[l] = shoupFactor(kinv[l], p);

		// omega = psi_N^(2k), a primitive d-th root.
		const uint64_t omega  = modpow(psiN[l], 2ull * k, p);
		const uint64_t iomega = modinv(omega, p);
		uint64_t accf = 1, acci = 1;
		for (int m = 0; m < d; ++m) {
			wf[static_cast<size_t>(l) * d + m]	 = accf;
			wi[static_cast<size_t>(l) * d + m]	 = acci;
			wf_s[static_cast<size_t>(l) * d + m] = shoupFactor(accf, p);
			wi_s[static_cast<size_t>(l) * d + m] = shoupFactor(acci, p);
			accf = static_cast<uint64_t>((static_cast<__uint128_t>(accf) * omega) % p);
			acci = static_cast<uint64_t>((static_cast<__uint128_t>(acci) * iomega) % p);
		}

		// psi_N^(+-h0(s)*i), geometric in i so one modpow per slot suffices.
		// The inverse table folds in the 1/d of the inverse DFT.
		const uint64_t dinv = modinv(static_cast<uint64_t>(d) % p, p);
		for (int s = 0; s < k; ++s) {
			const uint64_t h0	= 2ull * bitrev(static_cast<uint32_t>(s), logk) + 1;
			const uint64_t bf	= modpow(psiN[l], h0, p);
			const uint64_t bi	= modinv(bf, p);
			uint64_t cf = 1, ci = dinv;
			for (int i = 0; i < d; ++i) {
				const size_t off = (static_cast<size_t>(l) * k + s) * d + i;
				pf[off]			 = cf;
				pi[off]			 = ci;
				cf = static_cast<uint64_t>((static_cast<__uint128_t>(cf) * bf) % p);
				ci = static_cast<uint64_t>((static_cast<__uint128_t>(ci) * bi) % p);
			}
		}
	}

	SubringTables t;
	t.k				 = k;
	t.d				 = d;
	t.numLimbs		 = numLimbs;
	t.psi			 = deviceCopy(psi);
	t.psi_shoup		 = deviceCopy(psi_s);
	t.psi_inv		 = deviceCopy(psi_i);
	t.psi_inv_shoup	 = deviceCopy(psi_i_s);
	t.kinv			 = deviceCopy(kinv);
	t.kinv_shoup	 = deviceCopy(kinv_s);
	t.primeids		 = deviceCopy(primeids);
	t.wfwd			 = deviceCopy(wf);
	t.wfwd_shoup	 = deviceCopy(wf_s);
	t.winv			 = deviceCopy(wi);
	t.winv_shoup	 = deviceCopy(wi_s);
	t.psiFwd		 = deviceCopy(pf);
	t.psiInv		 = deviceCopy(pi);
	CudaCheckErrorMod;

	return g_tables.emplace(key, t).first->second;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

constexpr int BM_TILE = 32;
constexpr int BM_TI	  = 4;
constexpr int BM_TJ	  = 4;

/**
 * Vec_k^d of Definition 2, fused with a transpose.
 *
 * Column j of the matrix encryption is an R_N element m_j whose entry (i,j) is
 * the R_k element carrying the coefficients m_j[i + d*t]. Reading along i is
 * contiguous in the source and writing along t is contiguous in the destination,
 * so the exchange goes through shared memory.
 */
__global__ void bm_gather(uint64_t* __restrict__ dst, const uint64_t* const* __restrict__ src, const int d, const int cols, const int k) {
	__shared__ uint64_t tile[BM_TILE][BM_TILE + 1];

	const int lc   = blockIdx.z;
	const int col  = lc % cols;
	const int limb = lc / cols;
	const uint64_t* __restrict__ s = src[lc];

	const int i0 = blockIdx.y * BM_TILE;
	const int t0 = blockIdx.x * BM_TILE;

	for (int r = threadIdx.y; r < BM_TILE; r += blockDim.y) {
		const int t = t0 + r;
		const int i = i0 + threadIdx.x;
		tile[r][threadIdx.x] = (t < k && i < d) ? s[i + static_cast<size_t>(d) * t] : 0;
	}
	__syncthreads();
	for (int r = threadIdx.y; r < BM_TILE; r += blockDim.y) {
		const int i = i0 + r;
		const int t = t0 + threadIdx.x;
		if (i < d && t < k)
			dst[((static_cast<size_t>(limb) * d + i) * cols + col) * k + t] = tile[threadIdx.x][r];
	}
}

/** Inverse of bm_gather: writes R_k entries back into R_N coefficient order. */
__global__ void bm_scatter(uint64_t* const* __restrict__ dst, const uint64_t* __restrict__ src, const int d, const int cols, const int k) {
	__shared__ uint64_t tile[BM_TILE][BM_TILE + 1];

	const int lc   = blockIdx.z;
	const int col  = lc % cols;
	const int limb = lc / cols;
	uint64_t* __restrict__ o = dst[lc];

	const int i0 = blockIdx.y * BM_TILE;
	const int t0 = blockIdx.x * BM_TILE;

	for (int r = threadIdx.y; r < BM_TILE; r += blockDim.y) {
		const int i = i0 + r;
		const int t = t0 + threadIdx.x;
		tile[r][threadIdx.x] = (i < d && t < k) ? src[((static_cast<size_t>(limb) * d + i) * cols + col) * k + t] : 0;
	}
	__syncthreads();
	for (int r = threadIdx.y; r < BM_TILE; r += blockDim.y) {
		const int t = t0 + r;
		const int i = i0 + threadIdx.x;
		if (t < k && i < d)
			o[i + static_cast<size_t>(d) * t] = tile[threadIdx.x][r];
	}
}

/**
 * Forward negacyclic NTT of length k, one transform per block.
 *
 * Output lands in bit-reversed order. Both operands of the GEMM go through this
 * same transform and the result is inverted by bm_intt_k, so the ordering never
 * becomes visible.
 */
__global__ void bm_ntt_k(uint64_t* __restrict__ data,
  const uint64_t* __restrict__ psi,
  const uint64_t* __restrict__ psi_shoup,
  const int* __restrict__ primeids,
  const int k,
  const int transformsPerLimb) {
	extern __shared__ uint64_t sh[];

	const int limb	 = blockIdx.y;
	const int pid	 = primeids[limb];
	const uint64_t p = C_.primes[pid];

	uint64_t* __restrict__ g			 = data + (static_cast<size_t>(limb) * transformsPerLimb + blockIdx.x) * k;
	const uint64_t* __restrict__ tw		 = psi + static_cast<size_t>(limb) * k;
	const uint64_t* __restrict__ tw_s	 = psi_shoup + static_cast<size_t>(limb) * k;

	for (int i = threadIdx.x; i < k; i += blockDim.x)
		sh[i] = g[i];
	__syncthreads();

	int t = k;
	for (int m = 1; m < k; m <<= 1) {
		t >>= 1;
		for (int b = threadIdx.x; b < (k >> 1); b += blockDim.x) {
			const int i	 = b / t;
			const int j	 = 2 * i * t + (b % t);
			const uint64_t U = sh[j];
			const uint64_t V = modmult<ALGO_SHOUP>(sh[j + t], tw[m + i], pid, tw_s[m + i]);
			uint64_t x = U + V;
			x		   = (x >= p) ? x - p : x;
			uint64_t y = U + p - V;
			y		   = (y >= p) ? y - p : y;
			sh[j]	   = x;
			sh[j + t]  = y;
		}
		__syncthreads();
	}

	for (int i = threadIdx.x; i < k; i += blockDim.x)
		g[i] = sh[i];
}

/** Inverse of bm_ntt_k; consumes bit-reversed input and restores natural order. */
__global__ void bm_intt_k(uint64_t* __restrict__ data,
  const uint64_t* __restrict__ psi_inv,
  const uint64_t* __restrict__ psi_inv_shoup,
  const uint64_t* __restrict__ kinv,
  const uint64_t* __restrict__ kinv_shoup,
  const int* __restrict__ primeids,
  const int k,
  const int transformsPerLimb) {
	extern __shared__ uint64_t sh[];

	const int limb	 = blockIdx.y;
	const int pid	 = primeids[limb];
	const uint64_t p = C_.primes[pid];

	uint64_t* __restrict__ g		  = data + (static_cast<size_t>(limb) * transformsPerLimb + blockIdx.x) * k;
	const uint64_t* __restrict__ tw	  = psi_inv + static_cast<size_t>(limb) * k;
	const uint64_t* __restrict__ tw_s = psi_inv_shoup + static_cast<size_t>(limb) * k;

	for (int i = threadIdx.x; i < k; i += blockDim.x)
		sh[i] = g[i];
	__syncthreads();

	int t = 1;
	for (int m = k; m > 1; m >>= 1) {
		const int h = m >> 1;
		for (int b = threadIdx.x; b < (k >> 1); b += blockDim.x) {
			const int i	 = b / t;
			const int j	 = 2 * i * t + (b % t);
			const uint64_t U = sh[j];
			const uint64_t V = sh[j + t];
			uint64_t x = U + V;
			x		   = (x >= p) ? x - p : x;
			uint64_t y = U + p - V;
			y		   = (y >= p) ? y - p : y;
			sh[j]	   = x;
			sh[j + t]  = modmult<ALGO_SHOUP>(y, tw[h + i], pid, tw_s[h + i]);
		}
		__syncthreads();
		t <<= 1;
	}

	const uint64_t ki  = kinv[limb];
	const uint64_t kis = kinv_shoup[limb];
	for (int i = threadIdx.x; i < k; i += blockDim.x)
		g[i] = modmult<ALGO_SHOUP>(sh[i], ki, pid, kis);
}

/**
 * FIDESlib's length-N NTT domain -> the R_k NTT domain, in one pass.
 *
 * Writing j = j_hi*d + j_lo, the length-N transform satisfies
 *     out[j] = m(psi^(2*brv_N(j)+1)),   brv_N(j) = brv_k(j_hi) + k*brv_d(j_lo)
 * so each j_hi is one residue class h0 = 2*brv_k(j_hi)+1 and, within it,
 *     out[j] = sum_i [m_i(zeta^h0) * psi^(h0*i)] * omega^(t*i),  t = brv_d(j_lo).
 * That is a plain cyclic d-point DFT, not a negacyclic one. Inverting just this
 * stage lands directly in the R_k domain: the INTT_k that a full INTT_N would
 * perform is exactly cancelled by the NTT_k that used to follow it.
 *
 * The class occupies d contiguous indices, so the load coalesces, and feeding
 * natural j_lo order into a Cooley-Tukey butterfly consumes the bit-reversal
 * for free.
 */
__global__ void bm_ntt_to_subring(uint64_t* __restrict__ dst,
  const uint64_t* const* __restrict__ src,
  const uint64_t* __restrict__ winv,
  const uint64_t* __restrict__ winv_shoup,
  const uint64_t* __restrict__ psiInv,
  const int* __restrict__ primeids,
  const int d,
  const int cols,
  const int k) {
	extern __shared__ uint64_t sh[];

	const int s	   = blockIdx.x;
	const int col  = blockIdx.y;
	const int limb = blockIdx.z;

	const int pid	 = primeids[limb];
	const uint64_t p = C_.primes[pid];

	const uint64_t* __restrict__ in	 = src[static_cast<size_t>(limb) * cols + col] + static_cast<size_t>(s) * d;
	const uint64_t* __restrict__ wt	 = winv + static_cast<size_t>(limb) * d;
	const uint64_t* __restrict__ wts = winv_shoup + static_cast<size_t>(limb) * d;
	const uint64_t* __restrict__ pt	 = psiInv + (static_cast<size_t>(limb) * k + s) * d;

	for (int u = threadIdx.x; u < d; u += blockDim.x)
		sh[u] = in[u];
	__syncthreads();

	// Cooley-Tukey: bit-reversed input, natural output.
	for (int len = 2; len <= d; len <<= 1) {
		const int half = len >> 1;
		const int step = d / len;
		for (int b = threadIdx.x; b < (d >> 1); b += blockDim.x) {
			const int blk  = b / half;
			const int j	   = b % half;
			const int base = blk * len;
			const uint64_t w = wt[step * j];
			const uint64_t u = sh[base + j];
			const uint64_t v = modmult<ALGO_SHOUP>(sh[base + j + half], w, pid, wts[step * j]);
			uint64_t x = u + v;
			x		   = (x >= p) ? x - p : x;
			uint64_t y = u + p - v;
			y		   = (y >= p) ? y - p : y;
			sh[base + j]		= x;
			sh[base + j + half] = y;
		}
		__syncthreads();
	}

	for (int i = threadIdx.x; i < d; i += blockDim.x)
		dst[((static_cast<size_t>(limb) * d + i) * cols + col) * k + s] = modmult<ALGO_BARRETT>(sh[i], pt[i], pid);
}

/** Inverse of bm_ntt_to_subring. */
__global__ void bm_subring_to_ntt(uint64_t* const* __restrict__ dst,
  const uint64_t* __restrict__ src,
  const uint64_t* __restrict__ wfwd,
  const uint64_t* __restrict__ wfwd_shoup,
  const uint64_t* __restrict__ psiFwd,
  const int* __restrict__ primeids,
  const int d,
  const int cols,
  const int k) {
	extern __shared__ uint64_t sh[];

	const int s	   = blockIdx.x;
	const int col  = blockIdx.y;
	const int limb = blockIdx.z;

	const int pid	 = primeids[limb];
	const uint64_t p = C_.primes[pid];

	uint64_t* __restrict__ out		 = dst[static_cast<size_t>(limb) * cols + col] + static_cast<size_t>(s) * d;
	const uint64_t* __restrict__ wt	 = wfwd + static_cast<size_t>(limb) * d;
	const uint64_t* __restrict__ wts = wfwd_shoup + static_cast<size_t>(limb) * d;
	const uint64_t* __restrict__ pt	 = psiFwd + (static_cast<size_t>(limb) * k + s) * d;

	for (int i = threadIdx.x; i < d; i += blockDim.x)
		sh[i] = modmult<ALGO_BARRETT>(src[((static_cast<size_t>(limb) * d + i) * cols + col) * k + s], pt[i], pid);
	__syncthreads();

	// Gentleman-Sande: natural input, bit-reversed output.
	for (int len = d; len >= 2; len >>= 1) {
		const int half = len >> 1;
		const int step = d / len;
		for (int b = threadIdx.x; b < (d >> 1); b += blockDim.x) {
			const int blk  = b / half;
			const int j	   = b % half;
			const int base = blk * len;
			const uint64_t u = sh[base + j];
			const uint64_t v = sh[base + j + half];
			uint64_t x = u + v;
			x		   = (x >= p) ? x - p : x;
			uint64_t y = u + p - v;
			y		   = (y >= p) ? y - p : y;
			sh[base + j]		= x;
			sh[base + j + half] = modmult<ALGO_SHOUP>(y, wt[step * j], pid, wts[step * j]);
		}
		__syncthreads();
	}

	for (int u = threadIdx.x; u < d; u += blockDim.x)
		out[u] = sh[u];
}

/** Shoup precomputation for the fixed plaintext operand of the GEMM. */
__global__ void bm_shoup(uint64_t* __restrict__ shoup, const uint64_t* __restrict__ val, const int* __restrict__ primeids, const size_t perLimb) {
	const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i >= perLimb)
		return;
	const int limb	 = blockIdx.y;
	const uint64_t p = C_.primes[primeids[limb]];
	const size_t off = static_cast<size_t>(limb) * perLimb + i;
	shoup[off]		 = static_cast<uint64_t>((static_cast<__uint128_t>(val[off]) << 64) / p);
}

/**
 * The PPMM of Algorithm 1: a matrix product over R_{q,k}.
 *
 * In the transformed domain the product is k independent scalar matrix products,
 * one per NTT index. Because the NTT index is the fastest-varying dimension of
 * every operand, assigning it to threadIdx.x makes all loads and stores fully
 * coalesced, and each thread keeps a BM_TI x BM_TJ register tile to get reuse
 * along the two matrix dimensions.
 *
 * The plaintext operand carries Shoup factors, so each product costs three
 * multiplies and the accumulator stays reduced with a single conditional
 * subtraction.
 */
__global__ void bm_gemm(uint64_t* __restrict__ C,
  const uint64_t* __restrict__ A,
  const uint64_t* __restrict__ U,
  const uint64_t* __restrict__ Ushoup,
  const int* __restrict__ primeids,
  const int rowsA,
  const int inner,
  const int colsB,
  const int k) {
	const int s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= k)
		return;

	const int tilesJ = (colsB + BM_TJ - 1) / BM_TJ;
	const int i0	 = (blockIdx.y / tilesJ) * BM_TI;
	const int j0	 = (blockIdx.y % tilesJ) * BM_TJ;
	const int limb	 = blockIdx.z;
	const uint64_t p = C_.primes[primeids[limb]];

	const uint64_t* __restrict__ Al	 = A + static_cast<size_t>(limb) * rowsA * inner * k;
	const uint64_t* __restrict__ Ul	 = U + static_cast<size_t>(limb) * inner * colsB * k;
	const uint64_t* __restrict__ Usl = Ushoup + static_cast<size_t>(limb) * inner * colsB * k;
	uint64_t* __restrict__ Cl		 = C + static_cast<size_t>(limb) * rowsA * colsB * k;

	uint64_t acc[BM_TI][BM_TJ];
#pragma unroll
	for (int a = 0; a < BM_TI; ++a)
#pragma unroll
		for (int b = 0; b < BM_TJ; ++b)
			acc[a][b] = 0;

	for (int t = 0; t < inner; ++t) {
		uint64_t av[BM_TI];
#pragma unroll
		for (int a = 0; a < BM_TI; ++a) {
			const int i = i0 + a;
			av[a]		= (i < rowsA) ? Al[(static_cast<size_t>(i) * inner + t) * k + s] : 0;
		}
		uint64_t uv[BM_TJ], us[BM_TJ];
#pragma unroll
		for (int b = 0; b < BM_TJ; ++b) {
			const int j		 = j0 + b;
			const size_t off = (static_cast<size_t>(t) * colsB + j) * k + s;
			uv[b]			 = (j < colsB) ? Ul[off] : 0;
			us[b]			 = (j < colsB) ? Usl[off] : 0;
		}
#pragma unroll
		for (int a = 0; a < BM_TI; ++a)
#pragma unroll
			for (int b = 0; b < BM_TJ; ++b) {
				const uint64_t prod = Shoup_mult_64(av[a], uv[b], us[b], p);
				uint64_t x			= acc[a][b] + prod;
				acc[a][b]			= (x >= p) ? x - p : x;
			}
	}

#pragma unroll
	for (int a = 0; a < BM_TI; ++a) {
		const int i = i0 + a;
		if (i >= rowsA)
			continue;
#pragma unroll
		for (int b = 0; b < BM_TJ; ++b) {
			const int j = j0 + b;
			if (j >= colsB)
				continue;
			Cl[(static_cast<size_t>(i) * colsB + j) * k + s] = acc[a][b];
		}
	}
}

/**
 * The PPMM of Algorithm 4, where both operands come from ciphertexts.
 *
 * Same structure as bm_gemm, but the right operand has no precomputed Shoup
 * factors (it changes every call), so products go through Barrett. Its two
 * strides let the caller consume it transposed without moving any data, which
 * is what the (.)^T in steps 1, 3 and 4 of Algorithm 4 asks for.
 */
__global__ void bm_gemm_barrett(uint64_t* __restrict__ C,
  const uint64_t* __restrict__ A,
  const uint64_t* __restrict__ B,
  const int* __restrict__ primeids,
  const int rowsA,
  const int inner,
  const int colsB,
  const int k,
  const int bStrideT,
  const int bStrideJ) {
	const int s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= k)
		return;

	const int tilesJ = (colsB + BM_TJ - 1) / BM_TJ;
	const int i0	 = (blockIdx.y / tilesJ) * BM_TI;
	const int j0	 = (blockIdx.y % tilesJ) * BM_TJ;
	const int limb	 = blockIdx.z;
	const int pid	 = primeids[limb];
	const uint64_t p = C_.primes[pid];

	const uint64_t* __restrict__ Al = A + static_cast<size_t>(limb) * rowsA * inner * k;
	const uint64_t* __restrict__ Bl = B + static_cast<size_t>(limb) * inner * colsB * k;
	uint64_t* __restrict__ Cl		= C + static_cast<size_t>(limb) * rowsA * colsB * k;

	uint64_t acc[BM_TI][BM_TJ];
#pragma unroll
	for (int a = 0; a < BM_TI; ++a)
#pragma unroll
		for (int b = 0; b < BM_TJ; ++b)
			acc[a][b] = 0;

	for (int t = 0; t < inner; ++t) {
		uint64_t av[BM_TI];
#pragma unroll
		for (int a = 0; a < BM_TI; ++a) {
			const int i = i0 + a;
			av[a]		= (i < rowsA) ? Al[(static_cast<size_t>(i) * inner + t) * k + s] : 0;
		}
		uint64_t bv[BM_TJ];
#pragma unroll
		for (int b = 0; b < BM_TJ; ++b) {
			const int j = j0 + b;
			bv[b]		= (j < colsB) ? Bl[(static_cast<size_t>(t) * bStrideT + static_cast<size_t>(j) * bStrideJ) * k + s] : 0;
		}
#pragma unroll
		for (int a = 0; a < BM_TI; ++a)
#pragma unroll
			for (int b = 0; b < BM_TJ; ++b) {
				const uint64_t prod = modmult<ALGO_BARRETT>(av[a], bv[b], pid);
				uint64_t x			= acc[a][b] + prod;
				acc[a][b]			= (x >= p) ? x - p : x;
			}
	}

#pragma unroll
	for (int a = 0; a < BM_TI; ++a) {
		const int i = i0 + a;
		if (i >= rowsA)
			continue;
#pragma unroll
		for (int b = 0; b < BM_TJ; ++b) {
			const int j = j0 + b;
			if (j >= colsB)
				continue;
			Cl[(static_cast<size_t>(i) * colsB + j) * k + s] = acc[a][b];
		}
	}
}

/**
 * Transpose a matrix held as d column ciphertexts, in the coefficient domain.
 *
 * Entry (i,j) of the matrix lives in coefficient i + d*t of column j, so the
 * transpose is a pure permutation of coefficients across ciphertexts and needs
 * no subring transform.
 */
__global__ void bm_transpose_columns(uint64_t* const* __restrict__ dst, const uint64_t* const* __restrict__ src, const int d, const int k) {
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= k)
		return;
	const int j	   = blockIdx.y;
	const int limb = blockIdx.z;

	uint64_t* __restrict__ o = dst[static_cast<size_t>(limb) * d + j];
	for (int i = 0; i < d; ++i)
		o[i + static_cast<size_t>(d) * t] = src[static_cast<size_t>(limb) * d + i][j + static_cast<size_t>(d) * t];
}

// ---------------------------------------------------------------------------
// Limb plumbing
// ---------------------------------------------------------------------------

/** Batch matrix operations run on a single device; FIDESlib may spread limbs across several. */
void requireSingleDevice(const ContextData& cc) {
	if (cc.GPUid.size() != 1)
		throw std::runtime_error("batch matrix multiplication currently requires a single-GPU context");
}

uint64_t* limbData(const RNSPoly& poly, int limb) {
	const auto& l = poly.GPU[0].limb[limb];
	if (l.index() != FIDESlib::TYPE::U64)
		throw std::runtime_error("batch matrix multiplication requires 64-bit RNS limbs");
	return std::get<FIDESlib::TYPE::U64>(l).v.data;
}

int limbPrimeId(const RNSPoly& poly, int limb) {
	const auto& l = poly.GPU[0].limb[limb];
	return std::get<FIDESlib::TYPE::U64>(l).primeid;
}

/** The 2N-th roots FIDESlib's own transform is built on, per limb. */
std::vector<uint64_t> rootsOfUnity(const ContextData& cc, const std::vector<int>& primeids) {
	if (!cc.param.raw.has_value())
		throw std::runtime_error("batch matrix multiplication needs the OpenFHE roots of unity; this context has no RawParams");
	std::vector<uint64_t> r(primeids.size());
	for (size_t i = 0; i < primeids.size(); ++i)
		r[i] = cc.param.raw->root_of_unity.at(primeids[i]);
	return r;
}

/** Ciphertext components in the length-N NTT domain -> subring tensor. */
void launchToSubring(uint64_t* dst, const uint64_t* const* src, const SubringTables& t, int d, int cols, int k, int numLimbs) {
	const int threads = std::min(d / 2, 256);
	const dim3 grid(static_cast<unsigned>(k), static_cast<unsigned>(cols), static_cast<unsigned>(numLimbs));
	const size_t shmem = static_cast<size_t>(d) * sizeof(uint64_t);
	bm_ntt_to_subring<<<grid, threads, shmem>>>(dst, src, t.winv, t.winv_shoup, t.psiInv, t.primeids, d, cols, k);
}

/** Subring tensor -> ciphertext components in the length-N NTT domain. */
void launchFromSubring(uint64_t* const* dst, const uint64_t* src, const SubringTables& t, int d, int cols, int k, int numLimbs) {
	const int threads = std::min(d / 2, 256);
	const dim3 grid(static_cast<unsigned>(k), static_cast<unsigned>(cols), static_cast<unsigned>(numLimbs));
	const size_t shmem = static_cast<size_t>(d) * sizeof(uint64_t);
	bm_subring_to_ntt<<<grid, threads, shmem>>>(dst, src, t.wfwd, t.wfwd_shoup, t.psiFwd, t.primeids, d, cols, k);
}

void launchNTTk(uint64_t* data, const SubringTables& t, int k, int transformsPerLimb, int numLimbs, bool inverse) {
	const int threads = std::min(k / 2, 256);
	const dim3 grid(transformsPerLimb, numLimbs);
	const size_t shmem = static_cast<size_t>(k) * sizeof(uint64_t);
	if (inverse)
		bm_intt_k<<<grid, threads, shmem>>>(data, t.psi_inv, t.psi_inv_shoup, t.kinv, t.kinv_shoup, t.primeids, k, transformsPerLimb);
	else
		bm_ntt_k<<<grid, threads, shmem>>>(data, t.psi, t.psi_shoup, t.primeids, k, transformsPerLimb);
}

} // namespace

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

BatchMatrixLayout::BatchMatrixLayout(int N_, int d_) : N(N_), d(d_) {
	if (N <= 0 || (N & (N - 1)) != 0)
		throw std::invalid_argument("ring degree must be a positive power of two");
	if (d <= 0 || (d & (d - 1)) != 0 || d > N)
		throw std::invalid_argument("matrix row count must be a power of two dividing the ring degree");
	k = N / d;
	if (k < 2)
		throw std::invalid_argument("subring degree must be at least 2; reduce the matrix row count");
	batch = k / 2;
}

// ---------------------------------------------------------------------------
// Batch matrix encoding (Definition 1)
// ---------------------------------------------------------------------------

BatchMatrixEncoder::BatchMatrixEncoder(int k) : k_(k) {
	if (k < 2 || (k & (k - 1)) != 0)
		throw std::invalid_argument("subring degree must be a power of two of at least 2");

	pow_.resize(static_cast<size_t>(k_ / 2) * k_);
	const double pi		= std::acos(-1.0);
	const uint64_t mod	= 2ull * static_cast<uint64_t>(k_);
	uint64_t g			= 1;
	for (int j = 0; j < k_ / 2; ++j) {
		for (int t = 0; t < k_; ++t) {
			const uint64_t e = (g * static_cast<uint64_t>(t)) % mod;
			pow_[static_cast<size_t>(j) * k_ + t] = std::polar(1.0, pi * static_cast<double>(e) / k_);
		}
		g = (g * 5) % mod;
	}
}

void BatchMatrixEncoder::Encode(const std::vector<std::vector<std::complex<double>>>& batch, int rows, int cols, double Delta, std::vector<int64_t>& out) const {
	const int nslots	 = k_ / 2;
	const size_t entries = static_cast<size_t>(rows) * cols;
	if (static_cast<int>(batch.size()) != nslots)
		throw std::invalid_argument("batch must hold exactly k/2 matrices");
	for (const auto& m : batch)
		if (m.size() != entries)
			throw std::invalid_argument("every matrix in the batch must have rows*cols entries");

	out.assign(entries * k_, 0);
	const double scale = 2.0 / k_;

	for (size_t e = 0; e < entries; ++e) {
		int64_t* dst = out.data() + e * k_;
		// Entries that are zero across the whole batch encode to zero; skipping
		// them keeps sparse operands cheap, since this transform is O(k^2).
		bool nonzero = false;
		for (int j = 0; j < nslots && !nonzero; ++j)
			nonzero = batch[j][e] != std::complex<double>(0.0, 0.0);
		if (!nonzero)
			continue;
		for (int t = 0; t < k_; ++t) {
			double acc = 0.0;
			for (int j = 0; j < nslots; ++j) {
				const std::complex<double>& w = pow_[static_cast<size_t>(j) * k_ + t];
				const std::complex<double>& z = batch[j][e];
				acc += z.real() * w.real() + z.imag() * w.imag();
			}
			dst[t] = std::llround(Delta * scale * acc);
		}
	}
}

void BatchMatrixEncoder::Decode(const std::vector<int64_t>& in, int rows, int cols, double Delta, std::vector<std::vector<std::complex<double>>>& batch) const {
	const int nslots	 = k_ / 2;
	const size_t entries = static_cast<size_t>(rows) * cols;
	if (in.size() != entries * static_cast<size_t>(k_))
		throw std::invalid_argument("encoded matrix has the wrong length");

	batch.assign(nslots, std::vector<std::complex<double>>(entries, std::complex<double>(0.0, 0.0)));
	for (size_t e = 0; e < entries; ++e) {
		const int64_t* src = in.data() + e * k_;
		for (int j = 0; j < nslots; ++j) {
			std::complex<double> acc(0.0, 0.0);
			for (int t = 0; t < k_; ++t)
				acc += static_cast<double>(src[t]) * pow_[static_cast<size_t>(j) * k_ + t];
			batch[j][e] = acc / Delta;
		}
	}
}

// ---------------------------------------------------------------------------
// Matrix encryption layout (Definition 2)
// ---------------------------------------------------------------------------

void BuildMatrixEncryptionCoefficients(const std::vector<int64_t>& coeffs, const BatchMatrixLayout& layout, int rows, int cols, std::vector<std::vector<int64_t>>& out) {
	if (rows != layout.d)
		throw std::invalid_argument("matrix encryption requires the row count to equal the module rank d");
	if (coeffs.size() != static_cast<size_t>(rows) * cols * layout.k)
		throw std::invalid_argument("encoded matrix has the wrong length");

	out.assign(cols, std::vector<int64_t>(layout.N, 0));
	for (int j = 0; j < cols; ++j)
		for (int i = 0; i < rows; ++i) {
			const int64_t* e = coeffs.data() + (static_cast<size_t>(i) * cols + j) * layout.k;
			for (int t = 0; t < layout.k; ++t)
				out[j][i + static_cast<size_t>(layout.d) * t] = e[t];
		}
}

void SplitMatrixEncryptionCoefficients(const std::vector<std::vector<int64_t>>& in, const BatchMatrixLayout& layout, int rows, int cols, std::vector<int64_t>& coeffs) {
	if (rows != layout.d)
		throw std::invalid_argument("matrix encryption requires the row count to equal the module rank d");
	if (static_cast<int>(in.size()) != cols)
		throw std::invalid_argument("expected one coefficient vector per column");

	coeffs.assign(static_cast<size_t>(rows) * cols * layout.k, 0);
	for (int j = 0; j < cols; ++j) {
		if (in[j].size() != static_cast<size_t>(layout.N))
			throw std::invalid_argument("coefficient vectors must have length N");
		for (int i = 0; i < rows; ++i) {
			int64_t* e = coeffs.data() + (static_cast<size_t>(i) * cols + j) * layout.k;
			for (int t = 0; t < layout.k; ++t)
				e[t] = in[j][i + static_cast<size_t>(layout.d) * t];
		}
	}
}

// ---------------------------------------------------------------------------
// Plaintext matrix
// ---------------------------------------------------------------------------

BatchMatrixPlaintext::BatchMatrixPlaintext(Context& cc, const BatchMatrixLayout& layout, int rows, int cols, int level)
	: cc_(cc), layout_(layout), rows_(rows), cols_(cols), level_(level) {
	ContextData& data = *cc;
	requireSingleDevice(data);
	device_ = data.GPUid[0];

	const int numLimbs = level + 1;
	primeids_.resize(numLimbs);
	for (int l = 0; l < numLimbs; ++l)
		primeids_[l] = data.meta[0][l].id;

	const size_t elems = static_cast<size_t>(numLimbs) * rows_ * cols_ * layout_.k;
	cudaSetDevice(device_);
	cudaMalloc(&dev_, elems * sizeof(uint64_t));
	cudaMalloc(&dev_shoup_, elems * sizeof(uint64_t));
	CudaCheckErrorMod;
}

BatchMatrixPlaintext::BatchMatrixPlaintext(BatchMatrixPlaintext&& o) noexcept
	: NoiseFactor(o.NoiseFactor), cc_(o.cc_), layout_(o.layout_), rows_(o.rows_), cols_(o.cols_), level_(o.level_), device_(o.device_), primeids_(std::move(o.primeids_)),
	  dev_(o.dev_), dev_shoup_(o.dev_shoup_) {
	o.dev_		 = nullptr;
	o.dev_shoup_ = nullptr;
}

BatchMatrixPlaintext::~BatchMatrixPlaintext() {
	if (dev_)
		cudaFree(dev_);
	if (dev_shoup_)
		cudaFree(dev_shoup_);
}

void BatchMatrixPlaintext::Load(const std::vector<int64_t>& coeffs) {
	ContextData& data  = *cc_;
	const int numLimbs = static_cast<int>(primeids_.size());
	const int k		   = layout_.k;
	const size_t perLimb = static_cast<size_t>(rows_) * cols_ * k;

	if (coeffs.size() != perLimb)
		throw std::invalid_argument("encoded plaintext matrix has the wrong length");

	cudaSetDevice(device_);

	std::vector<uint64_t> host(static_cast<size_t>(numLimbs) * perLimb);
	std::vector<uint64_t> primes(numLimbs);
	for (int l = 0; l < numLimbs; ++l) {
		const uint64_t p = data.prime[primeids_[l]].p;
		primes[l]		 = p;
		for (size_t i = 0; i < perLimb; ++i)
			host[static_cast<size_t>(l) * perLimb + i] = centeredToModular(coeffs[i], p);
	}
	cudaMemcpy(dev_, host.data(), host.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

	const SubringTables& t = getSubringTables(k, primeids_, primes, rootsOfUnity(data, primeids_), data.N, device_);
	launchNTTk(dev_, t, k, static_cast<int>(perLimb / k), numLimbs, false);

	const int threads = 256;
	const dim3 grid(static_cast<unsigned>((perLimb + threads - 1) / threads), numLimbs);
	bm_shoup<<<grid, threads>>>(dev_shoup_, dev_, t.primeids, perLimb);
	CudaCheckErrorMod;
}

// ---------------------------------------------------------------------------
// Batch CPMM (Algorithm 1)
// ---------------------------------------------------------------------------

void BatchCPMM(std::vector<Ciphertext>& out, const std::vector<Ciphertext*>& in, const BatchMatrixPlaintext& U, bool rescale) {
	CudaNvtxRange range("FIDESlib::CKKS::BatchCPMM");

	const BatchMatrixLayout& L = U.layout();
	const int d				   = L.d;
	const int k				   = L.k;
	const int inner			   = U.rows();
	const int colsOut		   = U.cols();

	if (static_cast<int>(in.size()) != inner)
		throw std::invalid_argument("number of input ciphertexts must equal the plaintext matrix row count");
	if (in.empty())
		throw std::invalid_argument("no input ciphertexts");

	Context& cc_	= in[0]->cc_;
	ContextData& cc = in[0]->cc;
	SetCurrentContext(cc_);
	requireSingleDevice(cc);
	if (cc.N != L.N)
		throw std::invalid_argument("layout ring degree does not match the context");

	const int level	   = in[0]->c0.getLevel();
	const int numLimbs = level + 1;
	if (level != U.level())
		throw std::invalid_argument("plaintext matrix level does not match the ciphertexts");
	for (const Ciphertext* c : in)
		if (c->c0.getLevel() != level)
			throw std::invalid_argument("all input ciphertexts must share a level");

	const int device = cc.GPUid[0];
	cudaSetDevice(device);

	std::vector<int> primeids(numLimbs);
	std::vector<uint64_t> primes(numLimbs);
	for (int l = 0; l < numLimbs; ++l) {
		primeids[l] = limbPrimeId(in[0]->c0, l);
		primes[l]	= cc.prime[primeids[l]].p;
	}
	const SubringTables& tables = getSubringTables(k, primeids, primes, rootsOfUnity(cc, primeids), cc.N, device);

	// Outputs inherit shape and level from the inputs; their contents are fully
	// overwritten by the scatter below.
	out.clear();
	out.reserve(colsOut);
	for (int j = 0; j < colsOut; ++j) {
		out.emplace_back(cc_);
		out.back().copy(*in[0]);
	}

	// Touch every limb we will need before allocating anything: limbData rejects
	// 32-bit limbs by throwing, and the device buffers below are raw pointers
	// that an exception would leak.
	for (int l = 0; l < numLimbs; ++l)
		for (int j = 0; j < inner; ++j) {
			(void)limbData(in[j]->c0, l);
			(void)limbData(in[j]->c1, l);
		}
	for (int l = 0; l < numLimbs; ++l)
		for (int j = 0; j < colsOut; ++j) {
			(void)limbData(out[j].c0, l);
			(void)limbData(out[j].c1, l);
		}

	const size_t inElems  = static_cast<size_t>(numLimbs) * d * inner * k;
	const size_t outElems = static_cast<size_t>(numLimbs) * d * colsOut * k;

	// Stream-ordered allocation goes through the memory pool ContextData already
	// configures with an unlimited release threshold, so after the first call
	// these large buffers come back from the pool instead of the driver.
	uint64_t *A0 = nullptr, *A1 = nullptr, *C0 = nullptr, *C1 = nullptr;
	cudaMallocAsync(&A0, inElems * sizeof(uint64_t), 0);
	cudaMallocAsync(&A1, inElems * sizeof(uint64_t), 0);
	cudaMallocAsync(&C0, outElems * sizeof(uint64_t), 0);
	cudaMallocAsync(&C1, outElems * sizeof(uint64_t), 0);

	const uint64_t** devSrc0 = nullptr;
	const uint64_t** devSrc1 = nullptr;
	uint64_t** devDst0		 = nullptr;
	uint64_t** devDst1		 = nullptr;
	cudaMallocAsync(&devSrc0, static_cast<size_t>(numLimbs) * inner * sizeof(uint64_t*), 0);
	cudaMallocAsync(&devSrc1, static_cast<size_t>(numLimbs) * inner * sizeof(uint64_t*), 0);
	cudaMallocAsync(&devDst0, static_cast<size_t>(numLimbs) * colsOut * sizeof(uint64_t*), 0);
	cudaMallocAsync(&devDst1, static_cast<size_t>(numLimbs) * colsOut * sizeof(uint64_t*), 0);
	CudaCheckErrorMod;

	// The partial transform reads the inputs directly in the NTT domain, so they
	// are never modified and need no round trip.
	{
		std::vector<const uint64_t*> h0(static_cast<size_t>(numLimbs) * inner), h1(static_cast<size_t>(numLimbs) * inner);
		for (int l = 0; l < numLimbs; ++l)
			for (int j = 0; j < inner; ++j) {
				h0[static_cast<size_t>(l) * inner + j] = limbData(in[j]->c0, l);
				h1[static_cast<size_t>(l) * inner + j] = limbData(in[j]->c1, l);
			}
		cudaMemcpyAsync(devSrc0, h0.data(), h0.size() * sizeof(uint64_t*), cudaMemcpyHostToDevice, 0);
		cudaMemcpyAsync(devSrc1, h1.data(), h1.size() * sizeof(uint64_t*), cudaMemcpyHostToDevice, 0);

		std::vector<uint64_t*> g0(static_cast<size_t>(numLimbs) * colsOut), g1(static_cast<size_t>(numLimbs) * colsOut);
		for (int l = 0; l < numLimbs; ++l)
			for (int j = 0; j < colsOut; ++j) {
				g0[static_cast<size_t>(l) * colsOut + j] = limbData(out[j].c0, l);
				g1[static_cast<size_t>(l) * colsOut + j] = limbData(out[j].c1, l);
			}
		cudaMemcpyAsync(devDst0, g0.data(), g0.size() * sizeof(uint64_t*), cudaMemcpyHostToDevice, 0);
		cudaMemcpyAsync(devDst1, g1.data(), g1.size() * sizeof(uint64_t*), cudaMemcpyHostToDevice, 0);
		cudaStreamSynchronize(0); // host staging buffers go out of scope below
	}

	launchToSubring(A0, devSrc0, tables, d, inner, k, numLimbs);
	launchToSubring(A1, devSrc1, tables, d, inner, k, numLimbs);

	{
		const int threads = 32;
		const dim3 grid((k + threads - 1) / threads,
		  static_cast<unsigned>(((d + BM_TI - 1) / BM_TI) * ((colsOut + BM_TJ - 1) / BM_TJ)),
		  static_cast<unsigned>(numLimbs));
		bm_gemm<<<grid, threads>>>(C0, A0, U.data(), U.shoup(), tables.primeids, d, inner, colsOut, k);
		bm_gemm<<<grid, threads>>>(C1, A1, U.data(), U.shoup(), tables.primeids, d, inner, colsOut, k);
	}

	launchFromSubring(devDst0, C0, tables, d, colsOut, k, numLimbs);
	launchFromSubring(devDst1, C1, tables, d, colsOut, k, numLimbs);
	cudaDeviceSynchronize();
	CudaCheckErrorMod;

	cudaFreeAsync(A0, 0);
	cudaFreeAsync(A1, 0);
	cudaFreeAsync(C0, 0);
	cudaFreeAsync(C1, 0);
	cudaFreeAsync(devSrc0, 0);
	cudaFreeAsync(devSrc1, 0);
	cudaFreeAsync(devDst0, 0);
	cudaFreeAsync(devDst1, 0);
	CudaCheckErrorMod;

	// Step 2 of Algorithm 1: the product carries the plaintext scaling factor,
	// which the rescale removes.
	for (int j = 0; j < colsOut; ++j) {
		out[j].NoiseFactor = in[0]->NoiseFactor * U.NoiseFactor;
		out[j].NoiseLevel  = in[0]->NoiseLevel + 1;
		if (rescale)
			out[j].rescale();
	}
}

// ---------------------------------------------------------------------------
// Batch CMT and batch CCMM
// ---------------------------------------------------------------------------

namespace {

/** Owns a device array of per-(limb, column) limb pointers. */
struct DevicePointers {
	uint64_t** dev = nullptr;
	explicit DevicePointers(const std::vector<uint64_t*>& host) {
		cudaMallocAsync(&dev, host.size() * sizeof(uint64_t*), 0);
		cudaMemcpyAsync(dev, host.data(), host.size() * sizeof(uint64_t*), cudaMemcpyHostToDevice, 0);
	}
	~DevicePointers() {
		if (dev)
			cudaFreeAsync(dev, 0);
	}
	DevicePointers(const DevicePointers&)			 = delete;
	DevicePointers& operator=(const DevicePointers&) = delete;
};

std::vector<uint64_t*> componentPointers(const std::vector<Ciphertext*>& cts, bool useC1, int numLimbs) {
	const int cols = static_cast<int>(cts.size());
	std::vector<uint64_t*> h(static_cast<size_t>(numLimbs) * cols);
	for (int l = 0; l < numLimbs; ++l)
		for (int j = 0; j < cols; ++j)
			h[static_cast<size_t>(l) * cols + j] = limbData(useC1 ? cts[j]->c1 : cts[j]->c0, l);
	return h;
}

/** Coefficient-domain ciphertext component -> subring tensor in the R_k NTT domain. */
void gatherToTensor(const std::vector<Ciphertext*>& cts, bool useC1, uint64_t* tensor, const SubringTables& tables, int d, int k, int numLimbs) {
	const int cols = static_cast<int>(cts.size());
	DevicePointers ptrs(componentPointers(cts, useC1, numLimbs));
	launchToSubring(tensor, const_cast<const uint64_t* const*>(ptrs.dev), tables, d, cols, k, numLimbs);
}

/** Inverse of gatherToTensor; leaves the ciphertext component in the NTT domain. */
void scatterFromTensor(uint64_t* tensor, const std::vector<Ciphertext*>& cts, bool useC1, const SubringTables& tables, int d, int k, int numLimbs) {
	const int cols = static_cast<int>(cts.size());
	DevicePointers ptrs(componentPointers(cts, useC1, numLimbs));
	launchFromSubring(ptrs.dev, tensor, tables, d, cols, k, numLimbs);
}

std::vector<Ciphertext*> rawPointers(std::vector<Ciphertext>& v) {
	std::vector<Ciphertext*> r;
	r.reserve(v.size());
	for (auto& c : v)
		r.push_back(&c);
	return r;
}

void intttAll(const std::vector<Ciphertext*>& cts) {
	for (Ciphertext* c : cts) {
		c->c0.INTT<ALGO_SHOUP>(1, false);
		c->c1.INTT<ALGO_SHOUP>(1, false);
	}
	cudaDeviceSynchronize();
}

void nttAll(const std::vector<Ciphertext*>& cts) {
	for (Ciphertext* c : cts) {
		c->c0.NTT<ALGO_SHOUP>(1, false);
		c->c1.NTT<ALGO_SHOUP>(1, false);
	}
	cudaDeviceSynchronize();
}

/**
 * Algorithm 2, on ciphertexts that the caller owns.
 *
 * The butterfly is done in place against a single reusable scratch ciphertext.
 * The obvious formulation, lo->add(e,o) and hi->sub(e,o), costs two copies per
 * butterfly because Ciphertext's three-operand add and sub are implemented as
 * copy-then-accumulate, and it constructs two ciphertexts each time. Writing
 * e += o in place and keeping e - o in the scratch halves the copies and
 * removes the per-butterfly allocation entirely; the result is placed by
 * swapping the owning pointers rather than by copying.
 *
 * One scratch is enough for the whole recursion: the sub-calls finish before
 * this frame's combine loop starts, and the loop itself is sequential.
 */
void tweakRecursive(std::vector<std::unique_ptr<Ciphertext>>& ct, int k, int sgn, int N, Context& cc_, std::unique_ptr<Ciphertext>& scratch) {
	const int d = static_cast<int>(ct.size());
	if (d <= 1)
		return;

	std::vector<std::unique_ptr<Ciphertext>> even, odd;
	even.reserve(d / 2);
	odd.reserve(d / 2);
	for (int j = 0; j < d / 2; ++j) {
		even.push_back(std::move(ct[2 * j]));
		odd.push_back(std::move(ct[2 * j + 1]));
	}

	tweakRecursive(even, 2 * k, sgn, N, cc_, scratch);
	tweakRecursive(odd, 2 * k, sgn, N, cc_, scratch);

	const long long twoN = 2ll * N;
	for (int j = 0; j < d / 2; ++j) {
		const long long e	  = 2ll * k * j * sgn;
		const int power		  = static_cast<int>(((e % twoN) + twoN) % twoN);
		if (power != 0)
			odd[j]->multMonomial(power);

		scratch->copy(*even[j]); // the only copy in the butterfly
		even[j]->add(*odd[j]);	 // even := e + o, in place
		scratch->sub(*odd[j]);	 // scratch := e - o, in place
		std::swap(odd[j], scratch);

		ct[j]		  = std::move(even[j]);
		ct[j + d / 2] = std::move(odd[j]);
	}
}

} // namespace

std::vector<int> GetBatchCMTRotationIndices(const BatchMatrixLayout& layout) {
	const int N			= layout.N;
	const int k			= layout.k;
	const int d			= layout.d;
	const uint64_t mod	= 2ull * static_cast<uint64_t>(N);
	const int half		= N / 2;

	// rotation index r corresponds to the Galois element 5^r mod 2N
	std::map<uint64_t, int> galoisToIndex;
	uint64_t g = 1;
	for (int r = 0; r < half; ++r) {
		galoisToIndex.emplace(g, r);
		g = (g * 5) % mod;
	}

	std::vector<int> res;
	for (int t = 0; t < d; ++t) {
		const uint64_t h = (2ull * k * t + 1) % mod;
		auto it			 = galoisToIndex.find(h);
		if (it == galoisToIndex.end())
			throw std::runtime_error("automorphism X -> X^(2kt+1) is not a slot rotation for this layout");
		if (it->second != 0)
			res.push_back(it->second);
	}
	return res;
}

void BatchTweak(std::vector<Ciphertext>& ct, const BatchMatrixLayout& layout, int sgn) {
	if (static_cast<int>(ct.size()) != layout.d)
		throw std::invalid_argument("BatchTweak expects exactly d ciphertexts");
	Context& cc_ = ct[0].cc_;
	SetCurrentContext(cc_);

	std::vector<std::unique_ptr<Ciphertext>> work;
	work.reserve(layout.d);
	for (int i = 0; i < layout.d; ++i) {
		work.push_back(std::make_unique<Ciphertext>(cc_));
		work.back()->copy(ct[i]);
	}

	auto scratch = std::make_unique<Ciphertext>(cc_);
	scratch->copy(ct[0]); // give the scratch a valid level and shape
	tweakRecursive(work, layout.k, sgn, layout.N, cc_, scratch);

	for (int i = 0; i < layout.d; ++i)
		ct[i].copy(*work[i]);
}

void BatchCMT(std::vector<Ciphertext>& ct, const BatchMatrixLayout& layout) {
	CudaNvtxRange range("FIDESlib::CKKS::BatchCMT");

	const int d = layout.d;
	const int k = layout.k;
	const int N = layout.N;
	if (static_cast<int>(ct.size()) != d)
		throw std::invalid_argument("BatchCMT expects exactly d ciphertexts");

	Context& cc_	= ct[0].cc_;
	ContextData& cc = ct[0].cc;
	SetCurrentContext(cc_);

	// Step 1: ct_i <- X^i * ct_i
	for (int i = 1; i < d; ++i)
		ct[i].multMonomial(i);

	// Move into a pointer working set once. Every stage below permutes or
	// rewrites pointers, so the two TWEAKs and the automorphism permutation all
	// run without copying ciphertexts; only entry and exit copy.
	std::vector<std::unique_ptr<Ciphertext>> work;
	work.reserve(d);
	for (int i = 0; i < d; ++i) {
		work.push_back(std::make_unique<Ciphertext>(cc_));
		work.back()->copy(ct[i]);
	}
	auto scratch = std::make_unique<Ciphertext>(cc_);
	scratch->copy(ct[0]);

	// Step 2
	tweakRecursive(work, k, +1, N, cc_, scratch);

	// Step 3: scale by d^-1, then permute and apply the automorphisms. The map
	// t -> t* is a bijection, so scaling every ciphertext once is equivalent to
	// the per-t scaling written in the algorithm.
	{
		std::vector<uint64_t> dinv(cc.prime.size(), 0);
		for (size_t i = 0; i < cc.prime.size(); ++i) {
			const uint64_t p = cc.prime[i].p;
			if (p > 1)
				dinv[i] = modinv(static_cast<uint64_t>(d) % p, p);
		}
		for (int i = 0; i < d; ++i) {
			work[i]->c0.multScalar(dinv);
			work[i]->c1.multScalar(dinv);
		}
	}

	const uint64_t mod = 2ull * static_cast<uint64_t>(N);
	std::map<uint64_t, int> galoisToIndex;
	{
		uint64_t g = 1;
		for (int r = 0; r < N / 2; ++r) {
			galoisToIndex.emplace(g, r);
			g = (g * 5) % mod;
		}
	}

	// t -> t* is a bijection, so the permutation is a pure pointer shuffle and
	// each automorphism then runs in place.
	std::vector<std::unique_ptr<Ciphertext>> permuted(d);
	std::vector<int> rotIndex(d, 0);
	for (int t = 0; t < d; ++t) {
		const uint64_t h	= (2ull * k * t + 1) % mod;
		const uint64_t hinv = modinvGeneric(h, mod);
		const int tstar		= static_cast<int>((hinv - 1) / (2ull * k));
		if (tstar < 0 || tstar >= d)
			throw std::runtime_error("inverse Galois element fell outside the CMT index range");
		auto it = galoisToIndex.find(h);
		if (it == galoisToIndex.end())
			throw std::runtime_error("automorphism X -> X^(2kt+1) is not a slot rotation for this layout");
		rotIndex[t] = it->second;
		permuted[t] = std::move(work[tstar]);
	}
	work = std::move(permuted);

	for (int t = 0; t < d; ++t) {
		if (rotIndex[t] == 0)
			continue;
		// rotate() folds the index through normalyzeIndex, which is the
		// identity only at full slot count.
		const int savedSlots = work[t]->slots;
		work[t]->slots		 = N / 2;
		work[t]->rotate(rotIndex[t]);
		work[t]->slots = savedSlots;
	}

	// Step 4
	tweakRecursive(work, k, -1, N, cc_, scratch);

	for (int i = 0; i < d; ++i)
		ct[i].copy(*work[i]);

	// Step 5: ct'_i <- X^-i * ct'_i
	for (int i = 1; i < d; ++i)
		ct[i].multMonomial(2 * N - i);
}

namespace {

/**
 * Body of batch CCMM.
 *
 * @param preB,preA When non-null, the left operand is already in the R_k NTT
 *        domain and its gather is skipped. RectangularCCMM multiplies one left
 *        operand against every block of the right one, so hoisting that gather
 *        out of the loop removes k/2 - 1 redundant round trips through the
 *        length-N transform.
 */
void batchCCMMImpl(std::vector<Ciphertext>& out,
  const std::vector<Ciphertext*>& a,
  const std::vector<Ciphertext*>& b,
  const BatchMatrixLayout& layout,
  bool rescale,
  uint64_t* preB,
  uint64_t* preA) {
	CudaNvtxRange range("FIDESlib::CKKS::BatchCCMM");

	const int d = layout.d;
	const int k = layout.k;
	if (static_cast<int>(a.size()) != d || static_cast<int>(b.size()) != d)
		throw std::invalid_argument("BatchCCMM expects exactly d ciphertexts per operand");

	Context& cc_	= a[0]->cc_;
	ContextData& cc = a[0]->cc;
	SetCurrentContext(cc_);
	requireSingleDevice(cc);
	if (cc.N != layout.N)
		throw std::invalid_argument("layout ring degree does not match the context");

	const int level	   = a[0]->c0.getLevel();
	const int numLimbs = level + 1;
	const int device   = cc.GPUid[0];
	cudaSetDevice(device);

	std::vector<int> primeids(numLimbs);
	std::vector<uint64_t> primes(numLimbs);
	for (int l = 0; l < numLimbs; ++l) {
		primeids[l] = limbPrimeId(a[0]->c0, l);
		primes[l]	= cc.prime[primeids[l]].p;
	}
	const SubringTables& tables = getSubringTables(k, primeids, primes, rootsOfUnity(cc, primeids), cc.N, device);

	// Step 1: right operand becomes a row-wise matrix encryption. The transpose
	// is applied by the GEMM strides below rather than by moving data.
	std::vector<Ciphertext> bcmt;
	bcmt.reserve(d);
	for (int j = 0; j < d; ++j) {
		bcmt.emplace_back(cc_);
		bcmt.back().copy(*b[j]);
	}
	BatchCMT(bcmt, layout);

	const size_t elems	 = static_cast<size_t>(numLimbs) * d * d * k;
	const bool ownsLeft	 = (preB == nullptr);
	uint64_t *B = preB, *A = preA, *Bo = nullptr, *Ao = nullptr;
	uint64_t *C00 = nullptr, *C01 = nullptr, *C10 = nullptr, *C11 = nullptr;
	if (ownsLeft) {
		cudaMallocAsync(&B, elems * sizeof(uint64_t), 0);
		cudaMallocAsync(&A, elems * sizeof(uint64_t), 0);
	}
	for (uint64_t** p : { &Bo, &Ao, &C00, &C01, &C10, &C11 })
		cudaMallocAsync(p, elems * sizeof(uint64_t), 0);
	CudaCheckErrorMod;

	std::vector<Ciphertext*> bcmtPtrs = rawPointers(bcmt);

	if (ownsLeft) {
		gatherToTensor(a, false, B, tables, d, k, numLimbs);
		gatherToTensor(a, true, A, tables, d, k, numLimbs);
	}
	gatherToTensor(bcmtPtrs, false, Bo, tables, d, k, numLimbs);
	gatherToTensor(bcmtPtrs, true, Ao, tables, d, k, numLimbs);

	// Step 2. The right operands are consumed transposed: entry (t,j) of Bo^T
	// is Bo[j][t], so the strides are swapped.
	{
		const int threads = 32;
		const dim3 grid((k + threads - 1) / threads, static_cast<unsigned>(((d + BM_TI - 1) / BM_TI) * ((d + BM_TJ - 1) / BM_TJ)), static_cast<unsigned>(numLimbs));
		bm_gemm_barrett<<<grid, threads>>>(C00, B, Bo, tables.primeids, d, d, d, k, 1, d);
		bm_gemm_barrett<<<grid, threads>>>(C01, B, Ao, tables.primeids, d, d, d, k, 1, d);
		bm_gemm_barrett<<<grid, threads>>>(C10, A, Bo, tables.primeids, d, d, d, k, 1, d);
		bm_gemm_barrett<<<grid, threads>>>(C11, A, Ao, tables.primeids, d, d, d, k, 1, d);
		cudaDeviceSynchronize();
		CudaCheckErrorMod;
	}

	// Steps 3 and 4: fold each half back into ciphertexts, transpose it, and
	// convert it from row-wise to column-wise with a CMT.
	auto toCiphertexts = [&](uint64_t* c0src, uint64_t* c1src, std::vector<Ciphertext>& dst) {
		dst.clear();
		dst.reserve(d);
		for (int j = 0; j < d; ++j) {
			dst.emplace_back(cc_);
			dst.back().copy(*a[0]);
		}
		std::vector<Ciphertext*> p = rawPointers(dst);
		scatterFromTensor(c0src, p, false, tables, d, k, numLimbs);
		scatterFromTensor(c1src, p, true, tables, d, k, numLimbs);
		cudaDeviceSynchronize();
		BatchCMT(dst, layout);

		// Transpose the resulting matrices, a pure coefficient permutation.
		std::vector<Ciphertext> tr;
		tr.reserve(d);
		for (int j = 0; j < d; ++j) {
			tr.emplace_back(cc_);
			tr.back().copy(dst[j]);
		}
		std::vector<Ciphertext*> tp = rawPointers(tr);
		std::vector<Ciphertext*> sp = rawPointers(dst);
		intttAll(sp);
		intttAll(tp);
		{
			const dim3 grid((k + 127) / 128, static_cast<unsigned>(d), static_cast<unsigned>(numLimbs));
			for (int comp = 0; comp < 2; ++comp) {
				DevicePointers s(componentPointers(sp, comp == 1, numLimbs));
				DevicePointers t(componentPointers(tp, comp == 1, numLimbs));
				bm_transpose_columns<<<grid, 128>>>(t.dev, const_cast<const uint64_t* const*>(s.dev), d, k);
				cudaDeviceSynchronize();
			}
		}
		nttAll(tp);
		for (int j = 0; j < d; ++j)
			dst[j].copy(tr[j]);
	};

	std::vector<Ciphertext> D01, D23;
	toCiphertexts(C00, C01, D01);
	toCiphertexts(C10, C11, D23);

	if (ownsLeft) {
		cudaFreeAsync(B, 0);
		cudaFreeAsync(A, 0);
	}
	for (uint64_t* p : { Bo, Ao, C00, C01, C10, C11 })
		cudaFreeAsync(p, 0);
	CudaCheckErrorMod;

	// Step 5: relinearise (0, D3).
	std::vector<Ciphertext> E;
	E.reserve(d);
	{
		std::vector<uint64_t> zero(cc.prime.size(), 0);
		const KeySwitchingKey& relin = cc.GetEvalKey(a[0]->keyID);
		for (int j = 0; j < d; ++j) {
			E.emplace_back(cc_);
			E.back().copy(D23[j]);
			E.back().c0.multScalar(zero); // (0, D3)
			E.back().keySwitch(relin);
		}
	}

	// Step 6: Bres = D0 + E0, Ares = D1 + D2 + E1.
	out.clear();
	out.reserve(d);
	for (int j = 0; j < d; ++j) {
		out.emplace_back(cc_);
		out.back().copy(D01[j]);
		out.back().add(E[j]);
		out.back().c1.add(D23[j].c0);
	}
	cudaDeviceSynchronize();
	CudaCheckErrorMod;

	// Step 7
	for (int j = 0; j < d; ++j) {
		out[j].NoiseFactor = a[0]->NoiseFactor * b[0]->NoiseFactor;
		out[j].NoiseLevel  = a[0]->NoiseLevel + b[0]->NoiseLevel;
		if (rescale)
			out[j].rescale();
	}
}

} // namespace

void BatchCCMM(std::vector<Ciphertext>& out,
  const std::vector<Ciphertext*>& a,
  const std::vector<Ciphertext*>& b,
  const BatchMatrixLayout& layout,
  bool rescale) {
	batchCCMMImpl(out, a, b, layout, rescale, nullptr, nullptr);
}

// ---------------------------------------------------------------------------
// Rectangular matrix multiplication
// ---------------------------------------------------------------------------

namespace {

/**
 * The summation of Theorem 3.
 *
 * After a batch product the k/2 slot-wise results still sit side by side; their
 * sum appears as the constant term of each R_k entry. Reading the batch matrix
 * with k = 2 instead of k turns those constant terms into the top d rows, so a
 * CMT at that layout followed by a truncation extracts them, and a second CMT
 * returns the result to the original column-wise layout.
 *
 * Consumes N/2 ciphertexts and leaves d.
 */
void summationWithEncodingConversion(std::vector<Ciphertext>& v, const BatchMatrixLayout& layout) {
	const BatchMatrixLayout half(layout.N, layout.N / 2);
	BatchCMT(v, half);
	// Truncate to the top d rows. pop_back only destroys, whereas resize would
	// instantiate the default-construct path and Ciphertext has no default ctor.
	while (static_cast<int>(v.size()) > layout.d)
		v.pop_back();
	BatchCMT(v, layout);
}

} // namespace

std::vector<int> GetRectangularRotationIndices(const BatchMatrixLayout& layout) {
	std::vector<int> res = GetBatchCMTRotationIndices(layout);
	const BatchMatrixLayout half(layout.N, layout.N / 2);
	const std::vector<int> other = GetBatchCMTRotationIndices(half);
	res.insert(res.end(), other.begin(), other.end());
	std::sort(res.begin(), res.end());
	res.erase(std::unique(res.begin(), res.end()), res.end());
	return res;
}

void RectangularCPMM(std::vector<Ciphertext>& out, const std::vector<Ciphertext*>& in, const BatchMatrixPlaintext& U, const BatchMatrixLayout& layout) {
	CudaNvtxRange range("FIDESlib::CKKS::RectangularCPMM");

	if (static_cast<int>(in.size()) != layout.d)
		throw std::invalid_argument("RectangularCPMM expects exactly d ciphertexts");
	if (U.rows() != layout.d || U.cols() != layout.N / 2)
		throw std::invalid_argument("RectangularCPMM expects a d x N/2 plaintext matrix");

	// Step 1: the block products, scaled so the constant terms carry the sum
	// rather than the mean (Lemma 1 gives a factor of 2/k).
	BatchCPMM(out, in, U, /*rescale=*/true);
	for (Ciphertext& c : out)
		multIntScalar(c, static_cast<uint64_t>(layout.k / 2));

	// Steps 2 and 3
	summationWithEncodingConversion(out, layout);
}

void RectangularCCMM(std::vector<Ciphertext>& out, const std::vector<Ciphertext*>& in, const std::vector<Ciphertext*>& u, const BatchMatrixLayout& layout) {
	CudaNvtxRange range("FIDESlib::CKKS::RectangularCCMM");

	const int d	   = layout.d;
	const int half = layout.N / 2;
	if (static_cast<int>(in.size()) != d)
		throw std::invalid_argument("RectangularCCMM expects exactly d ciphertexts on the left");
	if (static_cast<int>(u.size()) != half)
		throw std::invalid_argument("RectangularCCMM expects N/2 ciphertexts on the right");
	if (half % d != 0)
		throw std::invalid_argument("N/2 must split into whole d-column blocks");

	Context& cc_	= in[0]->cc_;
	ContextData& cc = in[0]->cc;
	SetCurrentContext(cc_);
	requireSingleDevice(cc);

	// The left operand is the same for every block, so gather it once. Doing it
	// inside the loop would repeat a full length-N transform round trip on all
	// d ciphertexts for each of the k/2 blocks.
	const int k			= layout.k;
	const int level		= in[0]->c0.getLevel();
	const int numLimbs	= level + 1;
	const int device	= cc.GPUid[0];
	cudaSetDevice(device);

	std::vector<int> primeids(numLimbs);
	std::vector<uint64_t> primes(numLimbs);
	for (int l = 0; l < numLimbs; ++l) {
		primeids[l] = limbPrimeId(in[0]->c0, l);
		primes[l]	= cc.prime[primeids[l]].p;
	}
	const SubringTables& tables = getSubringTables(k, primeids, primes, rootsOfUnity(cc, primeids), cc.N, device);

	const size_t elems = static_cast<size_t>(numLimbs) * d * d * k;
	uint64_t *B = nullptr, *A = nullptr;
	cudaMallocAsync(&B, elems * sizeof(uint64_t), 0);
	cudaMallocAsync(&A, elems * sizeof(uint64_t), 0);
	CudaCheckErrorMod;

	gatherToTensor(in, false, B, tables, d, k, numLimbs);
	gatherToTensor(in, true, A, tables, d, k, numLimbs);

	out.clear();
	out.reserve(half);
	for (int blk = 0; blk < half / d; ++blk) {
		const std::vector<Ciphertext*> block(u.begin() + static_cast<size_t>(blk) * d, u.begin() + static_cast<size_t>(blk + 1) * d);
		std::vector<Ciphertext> part;
		batchCCMMImpl(part, in, block, layout, /*rescale=*/true, B, A);
		for (Ciphertext& c : part) {
			multIntScalar(c, static_cast<uint64_t>(layout.k / 2));
			out.emplace_back(cc_);
			out.back().copy(c);
		}
	}

	cudaFreeAsync(B, 0);
	cudaFreeAsync(A, 0);
	CudaCheckErrorMod;

	summationWithEncodingConversion(out, layout);
}

} // namespace FIDESlib::CKKS
