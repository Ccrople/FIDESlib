//
// Single-call Nsight Systems capture targets at the parameter set of
// Cheon, Kang and Lee, "Fast Batch Matrix Multiplication in Ciphertexts",
// Table 3, Method 1: log N = 16, 43-bit RNS moduli, level 37.
//
// Each test performs one warm-up invocation and then exactly one measured
// invocation bracketed by cudaProfilerStart/cudaProfilerStop, so a capture
// taken with
//
//     nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop
//
// contains the operation alone: context creation, key generation and the
// bootstrapping precomputation all happen outside the capture window and never
// reach the timeline.
//
// The paper fixes the ring degree, the limb width and the level but not the
// key-switching decomposition, so dnum is a free knob here; it, and every other
// parameter, can be overridden from the environment to retune a capture without
// rebuilding. Defaults are the paper's.
//
// Caveat on the 43-bit width, measured at logN=16 and level 37 with
// deg = firstMod - scaleMod held at 5: bootstrapping precision in this OpenFHE
// configuration is a smooth function of the scaling modulus, and 43 bits is
// below the usable end of it.
//
//     scaleModSize   44     46     48     50     52     55     59
//     worst error    18.8   3.65   0.823  0.407  0.097  0.0134  0.0026
//
// At 43 bits OpenFHE's Decode refuses to return at all. The level is not the
// problem -- level 37 at 59 bits bootstraps correctly (worst error 2.6e-3), and
// 43 bits fails at level 23 just as it does at 37, under both FIXEDAUTO and
// FLEXIBLEAUTOEXT. Reproducing the paper's precision at 43 bits would need
// composite scaling, which FIDESlib does not implement.
//
// The capture is still a valid performance measurement: RNS work depends on the
// number of limbs, not their bit width, so the timeline for a 43-bit run is the
// one a numerically sound run of the same shape would produce.
//

#include <openfhe.h>
#undef duration

#include "CKKS/BatchMatrix.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <gtest/gtest.h>
#include <exception>
#include <iostream>
#include <random>
#include <string>

namespace FIDESlib::Testing {

namespace {

/** Table 3, Method 1. The special modulus P is left to OpenFHE, which sizes it
 *  from dnum and is free to exceed the 43-bit limb width of Q.
 *
 *  q0 is 48 bits rather than the 60 the rest of the suite uses, because
 *  bootstrapping caps it: OpenFHE forms deg = round(log2(q0/Delta)) and then
 *  correction = correctionFactor - deg on an unsigned type. correctionFactor is
 *  derived from the ring degree and slot count and lands on 7 here, so a 60-bit
 *  q0 against a 43-bit Delta gives deg = 17 and wraps the subtraction. Upstream
 *  OpenFHE rejects deg > correctionFactor outright; FIDESlib carries that check
 *  commented out, so the wrap is silent and only shows up as a bootstrap whose
 *  output no longer decodes. 48 bits keeps deg = 5 and leaves the message range
 *  q0/Delta = 32, which comfortably covers the test values. */
constexpr int kPaperLogN		= 16;
constexpr int kPaperLevel		= 37;
constexpr int kPaperScaleMod	= 43;
constexpr int kPaperFirstMod	= 48;
constexpr int kPaperDnum		= 4;
/** Matrix dimension of the profiled batch product; k = N/d is the batch size. */
constexpr int kPaperMatrixDim	= 1024;

int EnvInt(const char* name, int fallback) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0')
		return fallback;
	return std::atoi(v);
}

lbcrypto::ScalingTechnique EnvScalingTechnique(lbcrypto::ScalingTechnique fallback) {
	const char* v = std::getenv("FIDESLIB_PROFILE_SCALETECH");
	if (v == nullptr || *v == '\0')
		return fallback;
	const std::string s(v);
	if (s == "FIXEDMANUAL")
		return lbcrypto::ScalingTechnique::FIXEDMANUAL;
	if (s == "FIXEDAUTO")
		return lbcrypto::ScalingTechnique::FIXEDAUTO;
	if (s == "FLEXIBLEAUTO")
		return lbcrypto::ScalingTechnique::FLEXIBLEAUTO;
	if (s == "FLEXIBLEAUTOEXT")
		return lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT;
	ADD_FAILURE() << "unknown FIDESLIB_PROFILE_SCALETECH: " << s;
	return fallback;
}

struct PaperParams {
	int logN	 = kPaperLogN;
	int L		 = kPaperLevel;
	int scaleMod = kPaperScaleMod;
	int firstMod = kPaperFirstMod;
	int dnum	 = kPaperDnum;
	lbcrypto::ScalingTechnique scaleTech = lbcrypto::ScalingTechnique::FIXEDAUTO;

	static PaperParams FromEnv(lbcrypto::ScalingTechnique defaultTech) {
		PaperParams p;
		p.logN		= EnvInt("FIDESLIB_PROFILE_LOGN", kPaperLogN);
		p.L			= EnvInt("FIDESLIB_PROFILE_L", kPaperLevel);
		p.scaleMod	= EnvInt("FIDESLIB_PROFILE_SCALEMOD", kPaperScaleMod);
		p.firstMod	= EnvInt("FIDESLIB_PROFILE_FIRSTMOD", kPaperFirstMod);
		p.dnum		= EnvInt("FIDESLIB_PROFILE_DNUM", kPaperDnum);
		p.scaleTech = EnvScalingTechnique(defaultTech);
		return p;
	}

	void Print(const char* what) const {
		std::cout << "[profile] " << what << " logN=" << logN << " L=" << L << " scaleModSize=" << scaleMod << " firstModSize=" << firstMod << " dnum=" << dnum
				  << std::endl;
	}
};

/** Builds the OpenFHE context and its FIDESlib mirror at the paper parameters.
 *  Security is HEStd_NotSet because the level and limb width are dictated by the
 *  table rather than by OpenFHE's parameter selection, exactly as the rest of
 *  the suite does. */
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> BuildCryptoContext(const PaperParams& p) {
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(p.L);
	parameters.SetFirstModSize(p.firstMod);
	parameters.SetScalingModSize(p.scaleMod);
	parameters.SetBatchSize(8);
	parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
	parameters.SetRingDim(1 << p.logN);
	parameters.SetNumLargeDigits(p.dnum);
	parameters.SetScalingTechnique(p.scaleTech);
	parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
	parameters.SetPREMode(lbcrypto::INDCPA);
	return GenCryptoContext(parameters);
}

} // namespace

/**
 * One basic (full-slot, non-encapsulated) bootstrap.
 *
 * Mirrors BtsTimingTests.Regular, but at the paper's parameters and reduced to
 * a single captured invocation.
 */
TEST(PaperProfile, BasicBootstrap) {
	const PaperParams p = PaperParams::FromEnv(lbcrypto::ScalingTechnique::FIXEDAUTO);
	p.Print("bootstrap");

	CKKS::DeregisterAllContexts();

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = BuildCryptoContext(p);
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	FIDESlib::CKKS::Parameters fp{ .logN = p.logN, .L = p.L, .dnum = p.dnum, .primes = p64, .Sprimes = sp64 };
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
	FIDESlib::CKKS::Context GPUcc_		= FIDESlib::CKKS::GenCryptoContextGPU(fp.adaptTo(raw_param), std::vector<int>{ 0 });
	FIDESlib::CKKS::ContextData& GPUcc	= *GPUcc_;
	GPUcc.batch							= 128;

	const int numSlots = static_cast<int>(cc->GetRingDimension()) / 2;
	std::cout << "[profile] slots=" << numSlots << " limbs=" << GPUcc.L + 1 << " dnum=" << GPUcc.dnum << std::endl;

	lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys = cc->KeyGen();

	cc->EvalBootstrapSetup({ 3, 3 }, { 16, 16 }, numSlots, 0, true, false);
	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	std::vector<double> x1			  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1		  = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
	auto c1							  = cc->Encrypt(keys.publicKey, ptxt1);
	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	// Warm-up on a separate ciphertext: the first call pays for pool growth and
	// lazy precomputation, which would otherwise dominate the captured timeline.
	// It cannot share the measured ciphertext, because Bootstrap raises the
	// level in place and does not drop back to the bottom on entry -- measuring
	// the second call on the same operand would profile a mid-level ModRaise
	// instead of the bottom-level one a real pipeline performs.
	{
		FIDESlib::CKKS::Ciphertext warm(GPUcc_, raw);
		FIDESlib::CKKS::Bootstrap(warm, numSlots, false);
		cudaDeviceSynchronize();
	}

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);
	cudaProfilerStart();
	FIDESlib::CKKS::Bootstrap(GPUct1, numSlots, false);
	cudaDeviceSynchronize();
	cudaProfilerStop();

	// Report numerical quality rather than assert on it. At the paper's 43-bit
	// width the bootstrap is known not to decode (see the table at the top), so
	// an assertion here would encode a known-bad configuration as a standing
	// suite failure; the capture is still wanted. Decode throws in that case,
	// hence the catch.
	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto result(c1);
	FIDESlib::CKKS::GetOpenFHECipherText(result, raw_res);
	try {
		lbcrypto::Plaintext result_pt;
		cc->Decrypt(keys.secretKey, result, &result_pt);
		double worst = 0.0;
		for (size_t i = 0; i < x1.size(); ++i) {
			const double got = result_pt->GetRealPackedValue().at(i);
			worst			 = std::max(worst, std::abs(got - x1[i]));
		}
		std::cout << "[profile] one Bootstrap done, level=" << GPUct1.getLevel() << " logPrecision=" << result_pt->GetLogPrecision()
				  << " worst error=" << worst << std::endl;
	} catch (const std::exception& e) {
		std::cout << "[profile] one Bootstrap done, level=" << GPUct1.getLevel() << " but the result does not decode: " << e.what() << std::endl;
	}

	CKKS::DeregisterAllContexts();
}

/**
 * One batch plaintext-ciphertext matrix multiplication (Algorithm 1) at the same
 * parameters, so its timeline is directly comparable with the bootstrap above.
 */
TEST(PaperProfile, PCMM) {
	const PaperParams p = PaperParams::FromEnv(lbcrypto::ScalingTechnique::FIXEDMANUAL);
	p.Print("PCMM");

	CKKS::DeregisterAllContexts();

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = BuildCryptoContext(p);
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::Parameters fp{ .logN = p.logN, .L = p.L, .dnum = p.dnum, .primes = p64, .Sprimes = sp64 };
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context cc_			= FIDESlib::CKKS::GenCryptoContextGPU(fp.adaptTo(raw_param), std::vector<int>{ 0 });
	FIDESlib::CKKS::ContextData& gpu	= *cc_;

	const int d = EnvInt("FIDESLIB_PROFILE_D", kPaperMatrixDim);
	const FIDESlib::CKKS::BatchMatrixLayout layout(gpu.N, d);
	const int k		= layout.k;
	const int inner = 8, cols = 8;

	std::mt19937 rng(11);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	std::uniform_int_distribution<int64_t> small(-4, 4);

	std::vector<FIDESlib::CKKS::Ciphertext> inputs;
	inputs.reserve(inner);
	for (int j = 0; j < inner; ++j) {
		std::vector<double> vals(8);
		for (auto& v : vals)
			v = dist(rng);
		lbcrypto::Plaintext pt			  = cc->MakeCKKSPackedPlaintext(vals);
		auto ct							  = cc->Encrypt(keys.publicKey, pt);
		FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, ct);
		inputs.emplace_back(cc_, raw);
	}
	const int level = inputs[0].c0.getLevel();

	std::vector<int64_t> U(static_cast<size_t>(inner) * cols * k);
	for (auto& v : U)
		v = small(rng);
	FIDESlib::CKKS::BatchMatrixPlaintext ptU(cc_, layout, inner, cols, level);
	ptU.NoiseFactor = inputs[0].NoiseFactor;
	ptU.Load(U);

	std::vector<FIDESlib::CKKS::Ciphertext*> in;
	for (auto& c : inputs)
		in.push_back(&c);

	// Warm-up, same reason as in the bootstrap capture.
	{
		std::vector<FIDESlib::CKKS::Ciphertext> warm;
		FIDESlib::CKKS::BatchCPMM(warm, in, ptU, /*rescale=*/false);
		cudaDeviceSynchronize();
	}

	std::vector<FIDESlib::CKKS::Ciphertext> out;
	cudaProfilerStart();
	FIDESlib::CKKS::BatchCPMM(out, in, ptU, /*rescale=*/false);
	cudaDeviceSynchronize();
	cudaProfilerStop();

	std::cout << "[profile] one BatchCPMM N=" << gpu.N << " d=" << d << " k=" << k << " inner=" << inner << " cols=" << cols << " limbs=" << level + 1
			  << std::endl;

	CKKS::DeregisterAllContexts();
}

} // namespace FIDESlib::Testing
