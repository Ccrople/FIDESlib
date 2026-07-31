//
// Tests for the batch matrix encoding, the matrix encryption layout and batch
// CPMM of Cheon, Kang and Lee, "Fast Batch Matrix Multiplication in Ciphertexts".
//

#include <openfhe.h>
#undef duration

#include "CKKS/BatchMatrix.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

#include <gtest/gtest.h>
#include <random>

namespace FIDESlib::Testing {

namespace {

/** Negacyclic product in Z_p[Y]/(Y^k + 1), accumulated into acc. */
void NegacyclicMulAcc(std::vector<uint64_t>& acc, const std::vector<uint64_t>& a, const std::vector<uint64_t>& b, uint64_t p) {
	const int k = static_cast<int>(a.size());
	for (int i = 0; i < k; ++i) {
		if (a[i] == 0)
			continue;
		for (int j = 0; j < k; ++j) {
			uint64_t v	= static_cast<uint64_t>((static_cast<__uint128_t>(a[i]) * b[j]) % p);
			int idx		= i + j;
			if (idx >= k) {
				idx -= k;
				v = (v == 0) ? 0 : p - v;
			}
			acc[idx] = (acc[idx] + v) % p;
		}
	}
}

uint64_t ModPow(uint64_t base, uint64_t exp, uint64_t p) {
	uint64_t r = 1, x = base % p;
	while (exp) {
		if (exp & 1)
			r = static_cast<uint64_t>((static_cast<__uint128_t>(r) * x) % p);
		x = static_cast<uint64_t>((static_cast<__uint128_t>(x) * x) % p);
		exp >>= 1;
	}
	return r;
}

uint32_t BitRev(uint32_t x, int bits) {
	uint32_t r = 0;
	for (int i = 0; i < bits; ++i) {
		r = (r << 1) | (x & 1u);
		x >>= 1;
	}
	return r;
}

int64_t ToCentered(uint64_t v, uint64_t p) {
	return (v > p / 2) ? static_cast<int64_t>(v) - static_cast<int64_t>(p) : static_cast<int64_t>(v);
}

uint64_t ToModular(int64_t v, uint64_t p) {
	if (v >= 0)
		return static_cast<uint64_t>(v) % p;
	const uint64_t m = static_cast<uint64_t>(-v) % p;
	return m == 0 ? 0 : p - m;
}

/** Naive negacyclic product over the integers, for the encoding tests. */
std::vector<int64_t> NegacyclicMulInt(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
	const int k = static_cast<int>(a.size());
	std::vector<int64_t> r(k, 0);
	for (int i = 0; i < k; ++i)
		for (int j = 0; j < k; ++j) {
			const int idx	 = i + j;
			const int64_t v	 = a[i] * b[j];
			if (idx >= k)
				r[idx - k] -= v;
			else
				r[idx] += v;
		}
	return r;
}

} // namespace

// ---------------------------------------------------------------------------
// Batch matrix encoding, Definition 1
// ---------------------------------------------------------------------------

TEST(BatchMatrixEncoding, RoundTrip) {
	constexpr int k	   = 32;
	constexpr int rows = 3;
	constexpr int cols = 2;
	const double Delta = std::pow(2.0, 30);

	FIDESlib::CKKS::BatchMatrixEncoder enc(k);
	std::mt19937 rng(12345);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	std::vector<std::vector<std::complex<double>>> batch(enc.slots(), std::vector<std::complex<double>>(rows * cols));
	for (auto& m : batch)
		for (auto& v : m)
			v = std::complex<double>(dist(rng), dist(rng));

	std::vector<int64_t> encoded;
	enc.Encode(batch, rows, cols, Delta, encoded);
	ASSERT_EQ(encoded.size(), static_cast<size_t>(rows) * cols * k);

	std::vector<std::vector<std::complex<double>>> decoded;
	enc.Decode(encoded, rows, cols, Delta, decoded);

	ASSERT_EQ(decoded.size(), batch.size());
	for (size_t l = 0; l < batch.size(); ++l)
		for (size_t e = 0; e < batch[l].size(); ++e) {
			EXPECT_NEAR(decoded[l][e].real(), batch[l][e].real(), 1e-6) << "slot " << l << " entry " << e;
			EXPECT_NEAR(decoded[l][e].imag(), batch[l][e].imag(), 1e-6) << "slot " << l << " entry " << e;
		}
}

/**
 * The property the whole construction rests on: multiplying two encoded entries
 * in R_k performs the entrywise product across every matrix in the batch at once.
 */
TEST(BatchMatrixEncoding, IsMultiplicative) {
	constexpr int k	   = 16;
	constexpr int rows = 2;
	constexpr int cols = 2;
	const double Delta = std::pow(2.0, 18);

	FIDESlib::CKKS::BatchMatrixEncoder enc(k);
	std::mt19937 rng(999);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	const size_t entries = rows * cols;
	std::vector<std::vector<std::complex<double>>> ma(enc.slots(), std::vector<std::complex<double>>(entries));
	std::vector<std::vector<std::complex<double>>> mb(enc.slots(), std::vector<std::complex<double>>(entries));
	for (int l = 0; l < enc.slots(); ++l)
		for (size_t e = 0; e < entries; ++e) {
			ma[l][e] = std::complex<double>(dist(rng), dist(rng));
			mb[l][e] = std::complex<double>(dist(rng), dist(rng));
		}

	std::vector<int64_t> ea, eb;
	enc.Encode(ma, rows, cols, Delta, ea);
	enc.Encode(mb, rows, cols, Delta, eb);

	// Entrywise product in R_k; the result carries Delta^2.
	std::vector<int64_t> prod(entries * k, 0);
	for (size_t e = 0; e < entries; ++e) {
		const std::vector<int64_t> a(ea.begin() + e * k, ea.begin() + (e + 1) * k);
		const std::vector<int64_t> b(eb.begin() + e * k, eb.begin() + (e + 1) * k);
		const std::vector<int64_t> c = NegacyclicMulInt(a, b);
		std::copy(c.begin(), c.end(), prod.begin() + e * k);
	}

	std::vector<std::vector<std::complex<double>>> decoded;
	enc.Decode(prod, rows, cols, Delta * Delta, decoded);

	for (int l = 0; l < enc.slots(); ++l)
		for (size_t e = 0; e < entries; ++e) {
			const std::complex<double> expected = ma[l][e] * mb[l][e];
			EXPECT_NEAR(decoded[l][e].real(), expected.real(), 1e-5) << "slot " << l << " entry " << e;
			EXPECT_NEAR(decoded[l][e].imag(), expected.imag(), 1e-5) << "slot " << l << " entry " << e;
		}
}

// ---------------------------------------------------------------------------
// Matrix encryption layout, Definition 2
// ---------------------------------------------------------------------------

TEST(BatchMatrixEncryption, CoefficientLayoutRoundTrip) {
	constexpr int N	   = 256;
	constexpr int d	   = 8;
	constexpr int cols = 3;
	const FIDESlib::CKKS::BatchMatrixLayout layout(N, d);
	ASSERT_EQ(layout.k, N / d);
	ASSERT_EQ(layout.batch, N / d / 2);

	std::vector<int64_t> coeffs(static_cast<size_t>(d) * cols * layout.k);
	std::mt19937 rng(7);
	std::uniform_int_distribution<int64_t> dist(-1000, 1000);
	for (auto& v : coeffs)
		v = dist(rng);

	std::vector<std::vector<int64_t>> columns;
	FIDESlib::CKKS::BuildMatrixEncryptionCoefficients(coeffs, layout, d, cols, columns);
	ASSERT_EQ(columns.size(), static_cast<size_t>(cols));
	for (const auto& c : columns)
		ASSERT_EQ(c.size(), static_cast<size_t>(N));

	std::vector<int64_t> back;
	FIDESlib::CKKS::SplitMatrixEncryptionCoefficients(columns, layout, d, cols, back);
	EXPECT_EQ(back, coeffs);
}

// ---------------------------------------------------------------------------
// Batch CPMM, Algorithm 1
// ---------------------------------------------------------------------------

/**
 * Runs batch CPMM on real ciphertexts and compares against a matrix product over
 * R_{q,k} evaluated on the host from the same limb data. The rescale of step 2
 * is skipped so the raw product is what gets compared.
 *
 * The contexts are built inside the test body rather than through the shared
 * parameterised fixture: the tparams globals in ParametrizedTest.cuh copy the
 * gparams globals during static initialisation, which is order-dependent across
 * translation units and can hand the fixture a partially zeroed parameter set.
 */
TEST(BatchMatrixTest, BatchCPMMMatchesReference) {
	// Matches the gparams64_13_2 family, the only configuration the existing
	// OpenFHE-backed suites exercise. Smaller ring degrees combined with
	// adaptTo() hit an unrelated pre-existing failure in context construction.
	constexpr int logN = 16;
	constexpr int L	   = 23;
	constexpr int dnum = 2;

	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(L);
	parameters.SetFirstModSize(60);
	parameters.SetScalingModSize(59);
	parameters.SetBatchSize(8);
	parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
	parameters.SetRingDim(1 << logN);
	parameters.SetNumLargeDigits(dnum);
	parameters.SetScalingTechnique(lbcrypto::ScalingTechnique::FIXEDMANUAL);
	parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
	parameters.SetPREMode(lbcrypto::INDCPA);

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys = cc->KeyGen();

	// Safe to read the shared prime tables here: static initialisation has
	// finished by the time a test body runs.
	FIDESlib::CKKS::Parameters fideslibParams{ .logN = logN, .L = L, .dnum = dnum, .primes = p64, .Sprimes = sp64 };

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context cc_			= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), std::vector<int>{ 0 });
	FIDESlib::CKKS::ContextData& gpu	= *cc_;

	const int N = gpu.N;
	// A large row count keeps the subring degree k small, which matters because
	// the host reference below is a naive O(k^2) negacyclic convolution.
	const int d = 1024;
	ASSERT_EQ(N % d, 0);
	const FIDESlib::CKKS::BatchMatrixLayout layout(N, d);
	const int k		  = layout.k;
	const int inner	  = 4;
	const int colsOut = 4;

	// Any valid ciphertexts will do: the test checks the linear-algebra pipeline,
	// which acts on the ciphertext polynomials regardless of what they encrypt.
	std::mt19937 rng(2024);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	std::vector<FIDESlib::CKKS::Ciphertext> inputs;
	inputs.reserve(inner);
	for (int j = 0; j < inner; ++j) {
		// The encrypted values are irrelevant here; only the ciphertext
		// polynomials matter, so a batch-sized vector is enough.
		std::vector<double> vals(8);
		for (auto& v : vals)
			v = dist(rng);
		lbcrypto::Plaintext pt = cc->MakeCKKSPackedPlaintext(vals);
		auto ct				   = cc->Encrypt(keys.publicKey, pt);
		FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, ct);
		inputs.emplace_back(cc_, raw);
	}

	const int level	   = inputs[0].c0.getLevel();
	const int numLimbs = level + 1;

	// Reference copy of the inputs in the coefficient domain.
	std::vector<std::vector<std::vector<uint64_t>>> refC0(inner), refC1(inner);
	for (int j = 0; j < inner; ++j) {
		FIDESlib::CKKS::Ciphertext tmp(cc_);
		tmp.copy(inputs[j]);
		tmp.c0.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		tmp.c1.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		cudaDeviceSynchronize();
		tmp.c0.store(refC0[j]);
		tmp.c1.store(refC1[j]);
	}

	// Plaintext matrix with small entries, taken directly as R_k coefficients so
	// that this test isolates the GPU pipeline from the encoder.
	std::uniform_int_distribution<int64_t> small(-4, 4);
	std::vector<int64_t> U(static_cast<size_t>(inner) * colsOut * k);
	for (auto& v : U)
		v = small(rng);

	FIDESlib::CKKS::BatchMatrixPlaintext ptU(cc_, layout, inner, colsOut, level);
	ptU.NoiseFactor = inputs[0].NoiseFactor;
	ptU.Load(U);

	std::vector<FIDESlib::CKKS::Ciphertext*> inPtrs;
	for (auto& c : inputs)
		inPtrs.push_back(&c);

	std::vector<FIDESlib::CKKS::Ciphertext> outputs;
	FIDESlib::CKKS::BatchCPMM(outputs, inPtrs, ptU, /*rescale=*/false);
	ASSERT_EQ(outputs.size(), static_cast<size_t>(colsOut));

	std::vector<std::vector<std::vector<uint64_t>>> gotC0(colsOut), gotC1(colsOut);
	for (int j = 0; j < colsOut; ++j) {
		outputs[j].c0.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		outputs[j].c1.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		cudaDeviceSynchronize();
		outputs[j].c0.store(gotC0[j]);
		outputs[j].c1.store(gotC1[j]);
	}

	// Rows are an independent tiling dimension of the GEMM, so a naive host
	// reference over every one of them would cost more than it proves. These
	// cover the first eight register tiles plus both boundaries.
	std::vector<int> checkRows;
	for (int i = 0; i < 32 && i < d; ++i)
		checkRows.push_back(i);
	checkRows.push_back(d / 2);
	checkRows.push_back(d - 1);

	// Host reference: (B * U, A * U) over R_{q,k}, one limb at a time.
	for (int comp = 0; comp < 2; ++comp) {
		const auto& ref = (comp == 0) ? refC0 : refC1;
		const auto& got = (comp == 0) ? gotC0 : gotC1;

		for (int l = 0; l < numLimbs; ++l) {
			const int primeid = gpu.meta[0][l].id;
			const uint64_t p  = gpu.prime[primeid].p;

			for (int i : checkRows) {
				for (int jj = 0; jj < colsOut; ++jj) {
					std::vector<uint64_t> acc(k, 0);
					for (int t = 0; t < inner; ++t) {
						std::vector<uint64_t> lhs(k), rhs(k);
						for (int s = 0; s < k; ++s) {
							lhs[s] = ref[t][l][i + static_cast<size_t>(d) * s];
							rhs[s] = ToModular(U[(static_cast<size_t>(t) * colsOut + jj) * k + s], p);
						}
						NegacyclicMulAcc(acc, lhs, rhs, p);
					}
					for (int s = 0; s < k; ++s) {
						const uint64_t expected = acc[s];
						const uint64_t actual	= got[jj][l][i + static_cast<size_t>(d) * s];
						ASSERT_EQ(actual, expected) << "component " << comp << " limb " << l << " row " << i << " col " << jj << " coeff " << s;
					}
				}
			}
		}
	}
}

/**
 * Runs batch CCMM and compares against a matrix product over R_{q,k} computed
 * on the host.
 *
 * Both operands are given a zero c1 component. That makes every key switch in
 * the pipeline exact — key switching a zero polynomial yields exactly zero, as
 * does relinearising (0,0) — so CMT, TWEAK, the automorphisms, the four GEMMs
 * and the final combination can all be checked for exact equality rather than
 * against a noise tolerance. With c1 = 0 the underlying matrix is simply the c0
 * matrix, so the expected result is the plain product of the two c0 matrices.
 */
TEST(BatchMatrixTest, BatchCCMMMatchesReference) {
	constexpr int logN = 16;
	constexpr int L	   = 23;
	constexpr int dnum = 2;

	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(L);
	parameters.SetFirstModSize(60);
	parameters.SetScalingModSize(59);
	parameters.SetBatchSize(8);
	parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
	parameters.SetRingDim(1 << logN);
	parameters.SetNumLargeDigits(dnum);
	parameters.SetScalingTechnique(lbcrypto::ScalingTechnique::FIXEDMANUAL);
	parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
	parameters.SetPREMode(lbcrypto::INDCPA);

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::Parameters fideslibParams{ .logN = logN, .L = L, .dnum = dnum, .primes = p64, .Sprimes = sp64 };

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context cc_			= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), std::vector<int>{ 0 });
	FIDESlib::CKKS::ContextData& gpu	= *cc_;

	const int d = 64;
	const FIDESlib::CKKS::BatchMatrixLayout layout(gpu.N, d);
	const int k = layout.k;

	// CMT needs the automorphisms X -> X^(2kt+1); CCMM additionally needs the
	// relinearisation key.
	FIDESlib::CKKS::GenAndAddRotationKeys(cc, keys, cc_, FIDESlib::CKKS::GetBatchCMTRotationIndices(layout));
	{
		FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
		FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
		kskEval.Initialize(rawKskEval);
		gpu.AddEvalKey(std::move(kskEval));
	}

	// A modest level keeps the host reference and the device tensors affordable.
	constexpr int testLevel = 3;
	const int numLimbs		= testLevel + 1;

	std::mt19937 rng(4242);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	std::vector<uint64_t> zeros(gpu.prime.size(), 0);

	auto makeOperand = [&](std::vector<FIDESlib::CKKS::Ciphertext>& dst) {
		dst.reserve(d);
		for (int j = 0; j < d; ++j) {
			std::vector<double> vals(8);
			for (auto& v : vals)
				v = dist(rng);
			lbcrypto::Plaintext pt			  = cc->MakeCKKSPackedPlaintext(vals);
			auto ct							  = cc->Encrypt(keys.publicKey, pt);
			FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, ct);
			dst.emplace_back(cc_, raw);
			dst.back().dropToLevel(testLevel);
			dst.back().c1.multScalar(zeros); // trivial encryption: c1 = 0
		}
	};

	std::vector<FIDESlib::CKKS::Ciphertext> opA, opB;
	makeOperand(opA);
	makeOperand(opB);

	// Snapshot the c0 matrices, which are the underlying matrices here.
	auto snapshot = [&](std::vector<FIDESlib::CKKS::Ciphertext>& src, std::vector<std::vector<std::vector<uint64_t>>>& dstC0) {
		dstC0.resize(d);
		for (int j = 0; j < d; ++j) {
			FIDESlib::CKKS::Ciphertext tmp(cc_);
			tmp.copy(src[j]);
			tmp.c0.INTT<FIDESlib::ALGO_SHOUP>(1, true);
			cudaDeviceSynchronize();
			tmp.c0.store(dstC0[j]);
		}
	};
	std::vector<std::vector<std::vector<uint64_t>>> refA, refB;
	snapshot(opA, refA);
	snapshot(opB, refB);

	std::vector<FIDESlib::CKKS::Ciphertext*> pa, pb;
	for (auto& c : opA)
		pa.push_back(&c);
	for (auto& c : opB)
		pb.push_back(&c);

	std::vector<FIDESlib::CKKS::Ciphertext> outputs;
	FIDESlib::CKKS::BatchCCMM(outputs, pa, pb, layout, /*rescale=*/false);
	ASSERT_EQ(outputs.size(), static_cast<size_t>(d));

	std::vector<std::vector<std::vector<uint64_t>>> gotC0(d), gotC1(d);
	for (int j = 0; j < d; ++j) {
		outputs[j].c0.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		outputs[j].c1.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		cudaDeviceSynchronize();
		outputs[j].c0.store(gotC0[j]);
		outputs[j].c1.store(gotC1[j]);
	}

	// The naive host reference costs d * k^2 per entry, so check a spread of
	// entries rather than all d^2 of them.
	std::vector<std::pair<int, int>> checks = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 3, 5 }, { d / 2, d / 2 }, { d - 1, d - 1 }, { d - 1, 0 }, { 0, d - 1 } };

	for (int l = 0; l < numLimbs; ++l) {
		const int primeid = gpu.meta[0][l].id;
		const uint64_t p  = gpu.prime[primeid].p;

		for (auto [i, jj] : checks) {
			std::vector<uint64_t> acc(k, 0);
			for (int t = 0; t < d; ++t) {
				std::vector<uint64_t> lhs(k), rhs(k);
				for (int s = 0; s < k; ++s) {
					lhs[s] = refA[t][l][i + static_cast<size_t>(d) * s];
					rhs[s] = refB[jj][l][t + static_cast<size_t>(d) * s];
				}
				NegacyclicMulAcc(acc, lhs, rhs, p);
			}
			for (int s = 0; s < k; ++s) {
				ASSERT_EQ(gotC0[jj][l][i + static_cast<size_t>(d) * s], acc[s]) << "c0 limb " << l << " row " << i << " col " << jj << " coeff " << s;
				ASSERT_EQ(gotC1[jj][l][i + static_cast<size_t>(d) * s], 0u) << "c1 should stay zero: limb " << l << " row " << i << " col " << jj;
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Rectangular matrix multiplication, Algorithms 5 and 6
// ---------------------------------------------------------------------------

namespace {

/** Shared setup for the rectangular tests. */
struct RectangularFixture {
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
	lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys;
	FIDESlib::CKKS::Context gpu_;
	int N = 0;

	void Build(int logN, int L, int dnum) {
		lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
		parameters.SetMultiplicativeDepth(L);
		parameters.SetFirstModSize(55);
		parameters.SetScalingModSize(45);
		parameters.SetBatchSize(8);
		parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
		parameters.SetRingDim(1 << logN);
		parameters.SetNumLargeDigits(dnum);
		parameters.SetScalingTechnique(lbcrypto::ScalingTechnique::FIXEDMANUAL);
		parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
		parameters.SetPREMode(lbcrypto::INDCPA);

		cc = GenCryptoContext(parameters);
		cc->Enable(lbcrypto::PKE);
		cc->Enable(lbcrypto::KEYSWITCH);
		cc->Enable(lbcrypto::LEVELEDSHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);

		FIDESlib::CKKS::Parameters fp{ .logN = logN, .L = L, .dnum = dnum, .primes = p64, .Sprimes = sp64 };
		FIDESlib::CKKS::RawParams raw = FIDESlib::CKKS::GetRawParams(cc);
		gpu_						  = FIDESlib::CKKS::GenCryptoContextGPU(fp.adaptTo(raw), std::vector<int>{ 0 });
		N							  = gpu_->N;
	}
};

/**
 * Build a trivial encryption: c1 = 0 and c0 set to the given coefficients.
 *
 * With c1 = 0 the decryption phase is exactly c0, so the ciphertext carries the
 * chosen polynomial exactly and every key switch downstream stays noiseless.
 */
FIDESlib::CKKS::Ciphertext MakeTrivial(FIDESlib::CKKS::Context& cc_,
  const FIDESlib::CKKS::Ciphertext& shape,
  const std::vector<int64_t>& coeffs,
  const std::vector<uint64_t>& zeros) {
	FIDESlib::CKKS::ContextData& gpu = *cc_;
	FIDESlib::CKKS::Ciphertext ct(cc_);
	ct.copy(shape);

	const int numLimbs = ct.c0.getLevel() + 1;
	std::vector<std::vector<uint64_t>> data(numLimbs, std::vector<uint64_t>(gpu.N, 0));
	std::vector<uint64_t> moduli(numLimbs);
	for (int l = 0; l < numLimbs; ++l) {
		const uint64_t p = gpu.prime[gpu.meta[0][l].id].p;
		moduli[l]		 = p;
		for (int i = 0; i < gpu.N; ++i)
			data[l][i] = ToModular(coeffs[i], p);
	}
	// load writes raw coefficients; the NTT then moves them to the evaluation
	// domain the rest of the library expects.
	ct.c0.load(data, moduli);
	ct.c0.NTT<FIDESlib::ALGO_SHOUP>(1, true);
	ct.c1.multScalar(const_cast<std::vector<uint64_t>&>(zeros));
	cudaDeviceSynchronize();
	return ct;
}

} // namespace

/**
 * Pins down what FIDESlib's length-N NTT actually computes.
 *
 * The partial-transform optimisation rests on the claim that the transform is
 * the textbook Cooley-Tukey negacyclic NTT with a bit-reversed twiddle table,
 * so that output[j] = m(psi^(2*brv(j)+1)) with psi the 2N-th root OpenFHE
 * supplies. Transforming m = X makes that directly checkable, since evaluating
 * X at a point returns the point itself.
 *
 * If this ever fails, the index algebra behind the partial transform is invalid
 * and must be re-derived rather than patched.
 */
TEST(BatchMatrixTest, NTTOrderingIsBitReversedNegacyclic) {
	RectangularFixture fx;
	fx.Build(/*logN=*/12, /*L=*/2, /*dnum=*/1);

	FIDESlib::CKKS::Context cc_		 = fx.gpu_;
	FIDESlib::CKKS::ContextData& gpu = *cc_;
	const int N						 = gpu.N;
	const int logN					 = gpu.logN;

	ASSERT_TRUE(gpu.param.raw.has_value()) << "need OpenFHE roots to know psi";

	std::vector<double> vals(8, 0.5);
	lbcrypto::Plaintext pt			  = fx.cc->MakeCKKSPackedPlaintext(vals);
	auto ct							  = fx.cc->Encrypt(fx.keys.publicKey, pt);
	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(fx.cc, ct);
	FIDESlib::CKKS::Ciphertext probe(cc_, raw);

	const int numLimbs = probe.c0.getLevel() + 1;
	std::vector<std::vector<uint64_t>> data(numLimbs, std::vector<uint64_t>(N, 0));
	std::vector<uint64_t> moduli(numLimbs);
	for (int l = 0; l < numLimbs; ++l) {
		moduli[l]  = gpu.prime[gpu.meta[0][l].id].p;
		data[l][1] = 1; // m(X) = X
	}
	probe.c0.load(data, moduli);
	probe.c0.NTT<FIDESlib::ALGO_SHOUP>(1, true);
	cudaDeviceSynchronize();

	std::vector<std::vector<uint64_t>> got;
	probe.c0.store(got);

	for (int l = 0; l < numLimbs; ++l) {
		const int primeid = gpu.meta[0][l].id;
		const uint64_t p  = gpu.prime[primeid].p;
		const uint64_t psi = gpu.param.raw->root_of_unity.at(primeid);

		// psi must be a primitive 2N-th root: psi^N == -1.
		ASSERT_EQ(ModPow(psi, N, p), p - 1) << "root_of_unity is not a 2N-th root for limb " << l;

		for (int j = 0; j < N; ++j) {
			const uint32_t r	  = BitRev(static_cast<uint32_t>(j), logN);
			const uint64_t expect = ModPow(psi, 2ull * r + 1, p);
			ASSERT_EQ(got[l][j], expect) << "NTT ordering differs at limb " << l << " index " << j;
		}
	}
}

/**
 * Rectangular CPMM against a directly computed W = M * U.
 *
 * U is made nonzero only in its first d columns, which keeps the O(k^2) host
 * encoder cheap and makes the expected answer a single d x d block: every
 * later block of W must come out zero. The k/2 batch slots are all populated,
 * so the summation of Theorem 3 is genuinely exercised.
 *
 * Note the output encoding: Theorem 3 yields M' = sum_j W_j Y^j, so block j of
 * W lands in the Y^j coefficient of each R_k entry, not in a batch slot.
 */
TEST(BatchMatrixTest, RectangularCPMMMatchesReference) {
	RectangularFixture fx;
	fx.Build(/*logN=*/12, /*L=*/4, /*dnum=*/1);

	FIDESlib::CKKS::Context cc_		 = fx.gpu_;
	FIDESlib::CKKS::ContextData& gpu = *cc_;
	const int N						 = fx.N;
	const int d						 = 64;
	const FIDESlib::CKKS::BatchMatrixLayout layout(N, d);
	const int k		= layout.k;
	const int slots = layout.batch;
	const int half	= N / 2;

	FIDESlib::CKKS::GenAndAddRotationKeys(fx.cc, fx.keys, cc_, FIDESlib::CKKS::GetRectangularRotationIndices(layout));

	std::mt19937 rng(31337);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	// M: slots blocks of d x d. U: slots blocks of d x half, nonzero only in
	// the leading d columns.
	std::vector<std::vector<std::complex<double>>> Mb(slots, std::vector<std::complex<double>>(static_cast<size_t>(d) * d));
	std::vector<std::vector<std::complex<double>>> Ub(slots, std::vector<std::complex<double>>(static_cast<size_t>(d) * half, { 0.0, 0.0 }));
	for (int l = 0; l < slots; ++l) {
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < d; ++j) {
				Mb[l][static_cast<size_t>(i) * d + j] = { dist(rng), 0.0 };
				Ub[l][static_cast<size_t>(i) * half + j] = { dist(rng), 0.0 };
			}
		}
	}

	// Expected leading block: W_0 = sum_l M_l * U_l[:, 0:d]
	std::vector<double> W0(static_cast<size_t>(d) * d, 0.0);
	for (int l = 0; l < slots; ++l)
		for (int i = 0; i < d; ++i)
			for (int j = 0; j < d; ++j) {
				double acc = 0.0;
				for (int t = 0; t < d; ++t)
					acc += Mb[l][static_cast<size_t>(i) * d + t].real() * Ub[l][static_cast<size_t>(t) * half + j].real();
				W0[static_cast<size_t>(i) * d + j] += acc;
			}

	FIDESlib::CKKS::BatchMatrixEncoder enc(k);
	const double Delta = std::pow(2.0, 45);

	std::vector<int64_t> Mcoeffs, Ucoeffs;
	enc.Encode(Mb, d, d, Delta, Mcoeffs);
	enc.Encode(Ub, d, half, Delta, Ucoeffs);

	std::vector<std::vector<int64_t>> Mcolumns;
	FIDESlib::CKKS::BuildMatrixEncryptionCoefficients(Mcoeffs, layout, d, d, Mcolumns);

	// A shape ciphertext to clone level and metadata from.
	std::vector<double> vals(8, 0.5);
	lbcrypto::Plaintext pt			  = fx.cc->MakeCKKSPackedPlaintext(vals);
	auto ctShape					  = fx.cc->Encrypt(fx.keys.publicKey, pt);
	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(fx.cc, ctShape);
	FIDESlib::CKKS::Ciphertext shape(cc_, raw);

	std::vector<uint64_t> zeros(gpu.prime.size(), 0);
	std::vector<FIDESlib::CKKS::Ciphertext> inputs;
	inputs.reserve(d);
	for (int j = 0; j < d; ++j)
		inputs.push_back(MakeTrivial(cc_, shape, Mcolumns[j], zeros));

	const int level = inputs[0].c0.getLevel();
	FIDESlib::CKKS::BatchMatrixPlaintext ptU(cc_, layout, d, half, level);
	ptU.NoiseFactor = Delta;
	ptU.Load(Ucoeffs);

	std::vector<FIDESlib::CKKS::Ciphertext*> in;
	for (auto& c : inputs)
		in.push_back(&c);

	std::vector<FIDESlib::CKKS::Ciphertext> out;
	FIDESlib::CKKS::RectangularCPMM(out, in, ptU, layout);
	ASSERT_EQ(out.size(), static_cast<size_t>(d));

	// c1 must still be exactly zero: nothing in the pipeline may inject noise
	// into a trivially encrypted operand.
	for (int j = 0; j < d; ++j) {
		std::vector<std::vector<uint64_t>> c1;
		out[j].c1.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		cudaDeviceSynchronize();
		out[j].c1.store(c1);
		for (size_t i = 0; i < c1[0].size(); ++i)
			ASSERT_EQ(c1[0][i], 0u) << "c1 leaked at column " << j << " coeff " << i;
	}

	const double scale = out[0].NoiseFactor;
	const uint64_t p0  = gpu.prime[gpu.meta[0][0].id].p;

	for (int j = 0; j < d; ++j) {
		std::vector<std::vector<uint64_t>> c0;
		out[j].c0.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		cudaDeviceSynchronize();
		out[j].c0.store(c0);

		for (int i = 0; i < d; ++i) {
			const double got	  = static_cast<double>(ToCentered(c0[0][i], p0)) / scale;
			const double expected = W0[static_cast<size_t>(i) * d + j];
			ASSERT_NEAR(got, expected, 1e-2 * std::max(1.0, std::abs(expected))) << "W0 at (" << i << "," << j << ")";
		}
		// Blocks beyond the first must vanish, since U is zero outside its
		// leading d columns.
		for (int t = 1; t < slots; ++t)
			for (int i = 0; i < d; ++i) {
				const double got = static_cast<double>(ToCentered(c0[0][i + static_cast<size_t>(d) * t], p0)) / scale;
				ASSERT_NEAR(got, 0.0, 1e-2) << "block " << t << " row " << i << " column " << j;
			}
	}
}

/**
 * Rectangular CCMM against the same directly computed W = M * U.
 *
 * Identical setup to the CPMM case except that U is encrypted, so the k/2
 * blocks go through batch CCMM rather than batch CPMM. Both operands are
 * trivially encrypted (c1 = 0), which keeps every key switch and the
 * relinearisations noiseless.
 */
TEST(BatchMatrixTest, RectangularCCMMMatchesReference) {
	RectangularFixture fx;
	fx.Build(/*logN=*/12, /*L=*/4, /*dnum=*/1);

	FIDESlib::CKKS::Context cc_		 = fx.gpu_;
	FIDESlib::CKKS::ContextData& gpu = *cc_;
	const int N						 = fx.N;
	const int d						 = 64;
	const FIDESlib::CKKS::BatchMatrixLayout layout(N, d);
	const int k		= layout.k;
	const int slots = layout.batch;
	const int half	= N / 2;

	FIDESlib::CKKS::GenAndAddRotationKeys(fx.cc, fx.keys, cc_, FIDESlib::CKKS::GetRectangularRotationIndices(layout));
	{
		FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
		FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(fx.keys);
		kskEval.Initialize(rawKskEval);
		gpu.AddEvalKey(std::move(kskEval));
	}

	std::mt19937 rng(90210);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	std::vector<std::vector<std::complex<double>>> Mb(slots, std::vector<std::complex<double>>(static_cast<size_t>(d) * d));
	std::vector<std::vector<std::complex<double>>> Ub(slots, std::vector<std::complex<double>>(static_cast<size_t>(d) * half, { 0.0, 0.0 }));
	for (int l = 0; l < slots; ++l)
		for (int i = 0; i < d; ++i)
			for (int j = 0; j < d; ++j) {
				Mb[l][static_cast<size_t>(i) * d + j]	 = { dist(rng), 0.0 };
				Ub[l][static_cast<size_t>(i) * half + j] = { dist(rng), 0.0 };
			}

	std::vector<double> W0(static_cast<size_t>(d) * d, 0.0);
	for (int l = 0; l < slots; ++l)
		for (int i = 0; i < d; ++i)
			for (int j = 0; j < d; ++j) {
				double acc = 0.0;
				for (int t = 0; t < d; ++t)
					acc += Mb[l][static_cast<size_t>(i) * d + t].real() * Ub[l][static_cast<size_t>(t) * half + j].real();
				W0[static_cast<size_t>(i) * d + j] += acc;
			}

	FIDESlib::CKKS::BatchMatrixEncoder enc(k);
	const double Delta = std::pow(2.0, 45);

	std::vector<int64_t> Mcoeffs, Ucoeffs;
	enc.Encode(Mb, d, d, Delta, Mcoeffs);
	enc.Encode(Ub, d, half, Delta, Ucoeffs);

	std::vector<std::vector<int64_t>> Mcolumns, Ucolumns;
	FIDESlib::CKKS::BuildMatrixEncryptionCoefficients(Mcoeffs, layout, d, d, Mcolumns);
	FIDESlib::CKKS::BuildMatrixEncryptionCoefficients(Ucoeffs, layout, d, half, Ucolumns);

	std::vector<double> vals(8, 0.5);
	lbcrypto::Plaintext pt			  = fx.cc->MakeCKKSPackedPlaintext(vals);
	auto ctShape					  = fx.cc->Encrypt(fx.keys.publicKey, pt);
	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(fx.cc, ctShape);
	FIDESlib::CKKS::Ciphertext shape(cc_, raw);

	std::vector<uint64_t> zeros(gpu.prime.size(), 0);
	std::vector<FIDESlib::CKKS::Ciphertext> inputs, uinputs;
	inputs.reserve(d);
	for (int j = 0; j < d; ++j)
		inputs.push_back(MakeTrivial(cc_, shape, Mcolumns[j], zeros));
	uinputs.reserve(half);
	for (int j = 0; j < half; ++j)
		uinputs.push_back(MakeTrivial(cc_, shape, Ucolumns[j], zeros));

	std::vector<FIDESlib::CKKS::Ciphertext*> in, u;
	for (auto& c : inputs)
		in.push_back(&c);
	for (auto& c : uinputs)
		u.push_back(&c);

	std::vector<FIDESlib::CKKS::Ciphertext> out;
	FIDESlib::CKKS::RectangularCCMM(out, in, u, layout);
	ASSERT_EQ(out.size(), static_cast<size_t>(d));

	const double scale = out[0].NoiseFactor;
	const uint64_t p0  = gpu.prime[gpu.meta[0][0].id].p;

	for (int j = 0; j < d; ++j) {
		std::vector<std::vector<uint64_t>> c0, c1;
		out[j].c0.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		out[j].c1.INTT<FIDESlib::ALGO_SHOUP>(1, true);
		cudaDeviceSynchronize();
		out[j].c0.store(c0);
		out[j].c1.store(c1);

		for (size_t i = 0; i < c1[0].size(); ++i)
			ASSERT_EQ(c1[0][i], 0u) << "c1 leaked at column " << j << " coeff " << i;

		for (int i = 0; i < d; ++i) {
			const double got	  = static_cast<double>(ToCentered(c0[0][i], p0)) / scale;
			const double expected = W0[static_cast<size_t>(i) * d + j];
			ASSERT_NEAR(got, expected, 1e-2 * std::max(1.0, std::abs(expected))) << "W0 at (" << i << "," << j << ")";
		}
		for (int t = 1; t < slots; ++t)
			for (int i = 0; i < d; ++i) {
				const double got = static_cast<double>(ToCentered(c0[0][i + static_cast<size_t>(d) * t], p0)) / scale;
				ASSERT_NEAR(got, 0.0, 1e-2) << "block " << t << " row " << i << " column " << j;
			}
	}
}

} // namespace FIDESlib::Testing
