
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace FIDESlib::CKKS;
using namespace std::chrono;

namespace FIDESlib::Testing {

class BtsTimingTests : public GeneralParametrizedTest {};

TEST_P(BtsTimingTests, Regular) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	// const bool sparse_encaps = false;

	std::cout << "Create context" << std::endl;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
	FIDESlib::CKKS::Context GPUcc_      = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *GPUcc_;
	std::cout << "Num large digits" << GPUcc.dnum << std::endl;
	// Parameters
	GPUcc.batch  = 128;
	int numSlots = cc->GetRingDimension() / 2;

	// Keys
	keys = cc->KeyGen();

	// Bootstrapping Precomputation

	cc->EvalBootstrapSetup({ 3, 3 }, { 16, 16 }, numSlots, 0, true, false);

	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
	std::cout << lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) << std::endl;
	std::cout << GPUcc.GetDoubleAngleIts() << std::endl;

	std::cout << "Add bootstrap precomputation" << std::endl;
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	std::vector<double> x1            = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1         = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
	auto c1                           = cc->Encrypt(keys.publicKey, ptxt1);
	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	std::cout << "Create ciphertext" << std::endl;
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);

	int N = 10;

	std::cout << "Begin boot" << std::endl;
	auto start_gpu = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		Bootstrap(GPUct1, numSlots, false);
		cudaDeviceSynchronize();
	}
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

	std::cout << GPUct1.getLevel() << std::endl;

	cudaDeviceSynchronize();

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto result(c1);
	GetOpenFHECipherText(result, raw_res);

	lbcrypto::Plaintext result_pt;
	cc->Decrypt(keys.secretKey, result, &result_pt);
	std::cout << result_pt->GetLogPrecision() << std::endl;
	for (int i = 0; i < 8; ++i) {
		std::cout << result_pt->GetRealPackedValue().at(i) << " ";
	}
	std::cout << std::endl;

	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
}

TEST_P(BtsTimingTests, SSE) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	// const bool sparse_encaps = true;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, ENCAPS);
	FIDESlib::CKKS::Context GPUcc_      = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *GPUcc_;

	// Parameters
	GPUcc.batch  = 128;
	int numSlots = cc->GetRingDimension() / 2;
	// int numSlots = 64;
	//  Keys
	keys = cc->KeyGen();

	// Bootstrapping Precomputation
	cc->EvalBootstrapSetup(
		{ 3, 3 },
		{ 16, 16 },
		numSlots,
		0,
		true,
		false,
		lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) + GPUcc.GetDoubleAngleIts());
	std::cout << lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) << std::endl;
	std::cout << GPUcc.GetDoubleAngleIts() << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	std::vector<double> x1    = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
	auto c1                   = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);

	GPUct1.dropToLevel(2);
	{
		FIDESlib::CKKS::RawCipherText raw_res;
		GPUct1.store(raw_res);
		auto result(c1);
		GetOpenFHECipherText(result, raw_res);

		lbcrypto::Plaintext result_pt;
		cc->Decrypt(keys.secretKey, result, &result_pt);
		std::cout << result_pt->GetLogPrecision() << std::endl;
	}

	int N = 10;

	auto start_gpu = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Ciphertext GPUct2(GPUcc_);
		cudaDeviceSynchronize();
		GPUct2.copy(GPUct1);
		cudaDeviceSynchronize();
		Bootstrap(GPUct2, numSlots, false);
		cudaDeviceSynchronize();
		GPUct1.copy(GPUct2);
		cudaDeviceSynchronize();
	}
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

	std::cout << GPUct1.getLevel() << std::endl;

	cudaDeviceSynchronize();

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto result(c1);
	GetOpenFHECipherText(result, raw_res);

	lbcrypto::Plaintext result_pt;
	cc->Decrypt(keys.secretKey, result, &result_pt);
	std::cout << result_pt->GetLogPrecision() << std::endl;
	for (int i = 0; i < 8; ++i) {
		std::cout << result_pt->GetRealPackedValue().at(i) << " ";
	}
	std::cout << std::endl;

	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
}

TEST_P(BtsTimingTests, REGULAR2) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	// const bool sparse_encaps = true;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM_2);
	FIDESlib::CKKS::Context GPUcc_      = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *GPUcc_;

	// Parameters
	GPUcc.batch  = 128;
	int numSlots = cc->GetRingDimension() / 2;

	// Keys
	keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(GPUcc_);
	eval_key_gpu.Initialize(eval_key);
	GPUcc.AddEvalKey(std::move(eval_key_gpu));

	// Bootstrapping Precomputation
	cc->EvalBootstrapSetup(
		{ 3, 3 },
		{ 16, 16 },
		numSlots,
		0,
		true,
		false,
		lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) + GPUcc.GetDoubleAngleIts());
	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	std::vector<double> x1    = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
	auto c1                   = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);

	GPUct1.dropToLevel(2);

	int N = 2;

	auto start_gpu = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		Bootstrap(GPUct1, numSlots, false);
		cudaDeviceSynchronize();
	}
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

	std::cout << GPUct1.getLevel() << std::endl;

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto result(c1);
	GetOpenFHECipherText(result, raw_res);

	lbcrypto::Plaintext result_pt;
	cc->Decrypt(keys.secretKey, result, &result_pt);
	std::cout << result_pt->GetLogPrecision() << std::endl;
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
}

TEST_P(BtsTimingTests, SPARSE) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	// const bool sparse_encaps = true;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, SPARSE);
	FIDESlib::CKKS::Context GPUcc_      = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *GPUcc_;

	// Parameters
	GPUcc.batch  = 128;
	int numSlots = cc->GetRingDimension() / 2;

	// Keys
	keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(GPUcc_);
	eval_key_gpu.Initialize(eval_key);
	GPUcc.AddEvalKey(std::move(eval_key_gpu));

	// Bootstrapping Precomputation
	cc->EvalBootstrapSetup(
		{ 3, 3 },
		{ 0, 0 },
		numSlots,
		0,
		true,
		false,
		lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) + GPUcc.GetDoubleAngleIts());
	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	std::vector<double> x1    = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
	auto c1                   = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);

	GPUct1.dropToLevel(2);

	int N = 10;

	auto start_gpu = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		Bootstrap(GPUct1, numSlots, false);
		cudaDeviceSynchronize();
	}
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

	std::cout << GPUct1.getLevel() << std::endl;

	cudaDeviceSynchronize();

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto result(c1);
	GetOpenFHECipherText(result, raw_res);

	lbcrypto::Plaintext result_pt;
	cc->Decrypt(keys.secretKey, result, &result_pt);
	std::cout << result_pt->GetLogPrecision() << std::endl;
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
}

INSTANTIATE_TEST_SUITE_P(LLMTests, BtsTimingTests, testing::Values(TTALL64BOOT));

//===----------------------------------------------------------------------===//
// How small can the modulus get and still bootstrap to a given precision?
//
// BootstrapPrecisionSweep.ScalingModulus runs one full-slot bootstrap per entry
// of a scaling-modulus list and reports, for each, the precision of the result
// and the total modulus the configuration actually consumed. Everything is set
// from the environment so a sweep can be retuned without rebuilding:
//
//   FIDESLIB_BPREC_LOGN         ring degree exponent            (default 12)
//   FIDESLIB_BPREC_SCALEMODS    comma-separated Delta widths    (default 40..59)
//   FIDESLIB_BPREC_DEG          firstMod - scaleMod             (default 1)
//   FIDESLIB_BPREC_LB0/_LB1     CoeffsToSlots/SlotsToCoeffs budget (default 3,3)
//   FIDESLIB_BPREC_DNUM         key-switching digits            (default 3)
//   FIDESLIB_BPREC_LEVELSAFTER  levels left after bootstrapping (default 1)
//   FIDESLIB_BPREC_SCALETECH    FIXEDAUTO | FLEXIBLEAUTO | ...  (default FIXEDAUTO)
//   FIDESLIB_BPREC_SLOTS        slot count, 0 means N/2         (default 0)
//   FIDESLIB_BPREC_CPU          also bootstrap on the CPU       (default 0)
//
// The multiplicative depth is not a free knob: it is derived from the level
// budget with OpenFHE's own GetBootstrapDepth, so each row is the *minimal*
// chain that can carry the requested bootstrap, plus LEVELSAFTER. That is what
// makes the reported logQP comparable against a security bound such as SEAL's
// CoeffModulus::MaxBitCount -- it is the cheapest modulus that does the job at
// that Delta, not an arbitrary one.
//
// deg = firstMod - scaleMod is a precision knob, not just a bookkeeping one.
// OpenFHE forms correction = correctionFactor - deg, where correctionFactor is
// the empirically fitted optimum for the ring degree and slot count, so every
// bit of deg spends a bit of that optimum. deg = 1 is what OpenFHE's own
// bootstrapping examples use and is the default here. deg above the correction
// factor wraps the unsigned subtraction outright; FIDESlib carries that guard
// commented out in src/CKKS/Bootstrap.cu, so the wrap is silent, and the test
// prints the correction factor next to deg to make the margin visible.
//
// q0 cannot exceed the 60-bit native word, so firstMod is clamped there and deg
// shrinks accordingly at the wide end of the sweep; the row reports the deg it
// actually used.
//===----------------------------------------------------------------------===//

namespace {

int BPrecEnvInt(const char* name, int fallback) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0')
		return fallback;
	return std::atoi(v);
}

std::vector<int> BPrecEnvIntList(const char* name, const std::vector<int>& fallback) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0')
		return fallback;
	std::vector<int> out;
	std::stringstream ss(v);
	std::string item;
	while (std::getline(ss, item, ',')) {
		if (!item.empty())
			out.push_back(std::atoi(item.c_str()));
	}
	return out.empty() ? fallback : out;
}

lbcrypto::ScalingTechnique BPrecEnvScalingTechnique(lbcrypto::ScalingTechnique fallback) {
	const char* v = std::getenv("FIDESLIB_BPREC_SCALETECH");
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
	ADD_FAILURE() << "unknown FIDESLIB_BPREC_SCALETECH: " << s;
	return fallback;
}

/** Sum of the bit lengths of a set of RNS moduli. This is the quantity a
 *  security bound constrains, so it is taken from the moduli OpenFHE actually
 *  generated rather than from the requested widths. */
double BPrecLogModulus(const std::vector<uint64_t>& moduli) {
	double bits = 0.0;
	for (uint64_t q : moduli)
		bits += std::log2(static_cast<double>(q));
	return bits;
}

/** Precision in bits of a bootstrapped result against its input, as
 *  -log2 of the error, for values drawn from [-1, 1]. Both the worst slot and
 *  the RMS over all slots are returned: the worst slot is the honest bound, the
 *  RMS is what a "typical" slot sees and is the closer analogue of the figure
 *  OpenFHE's GetLogPrecision reports. */
struct BPrecError {
	double worst;
	double rms;
};

BPrecError BPrecMeasure(const std::vector<double>& want, const std::vector<double>& got) {
	double worst = 0.0;
	double acc	 = 0.0;
	const size_t n = std::min(want.size(), got.size());
	for (size_t i = 0; i < n; ++i) {
		const double d = std::abs(got[i] - want[i]);
		worst		   = std::max(worst, d);
		acc += d * d;
	}
	return { worst, std::sqrt(acc / static_cast<double>(n)) };
}

double BPrecBits(double err) {
	return err > 0.0 ? -std::log2(err) : std::numeric_limits<double>::infinity();
}

} // namespace

TEST(BootstrapPrecisionSweep, ScalingModulus) {
	const int logN		  = BPrecEnvInt("FIDESLIB_BPREC_LOGN", 12);
	const int degWanted	  = BPrecEnvInt("FIDESLIB_BPREC_DEG", 1);
	const int lb0		  = BPrecEnvInt("FIDESLIB_BPREC_LB0", 3);
	const int lb1		  = BPrecEnvInt("FIDESLIB_BPREC_LB1", 3);
	const int dnum		  = BPrecEnvInt("FIDESLIB_BPREC_DNUM", 3);
	const int levelsAfter = BPrecEnvInt("FIDESLIB_BPREC_LEVELSAFTER", 1);
	const int slotsArg	  = BPrecEnvInt("FIDESLIB_BPREC_SLOTS", 0);
	const bool alsoCPU	  = BPrecEnvInt("FIDESLIB_BPREC_CPU", 0) != 0;
	const lbcrypto::ScalingTechnique scaleTech = BPrecEnvScalingTechnique(lbcrypto::ScalingTechnique::FIXEDAUTO);

	const std::vector<int> scaleMods = BPrecEnvIntList("FIDESLIB_BPREC_SCALEMODS", { 40, 44, 48, 52, 55, 59 });

	const std::vector<uint32_t> levelBudget{ static_cast<uint32_t>(lb0), static_cast<uint32_t>(lb1) };
	const uint32_t bootDepth = lbcrypto::FHECKKSRNS::GetBootstrapDepth(levelBudget, lbcrypto::UNIFORM_TERNARY);
	const int L				 = static_cast<int>(bootDepth) + levelsAfter;

	std::cout << "[bprec] logN=" << logN << " levelBudget={" << lb0 << "," << lb1 << "}"
			  << " bootstrapDepth=" << bootDepth << " levelsAfter=" << levelsAfter << " => L=" << L << " (limbs=" << L + 1 << ")"
			  << " dnum=" << dnum << " deg=" << degWanted << std::endl;

	// A native-word modulus cannot be wider than 60 bits, so a wide Delta eats
	// into deg rather than pushing q0 past the word.
	constexpr int kMaxModSize = 60;

	for (int scaleMod : scaleMods) {
		const int firstMod = std::min(scaleMod + degWanted, kMaxModSize);
		const int deg	   = firstMod - scaleMod;
		if (deg < 0) {
			std::cout << "[bprec] logN=" << logN << " scaleMod=" << scaleMod << " skipped: q0 would exceed " << kMaxModSize << " bits" << std::endl;
			continue;
		}

		CKKS::DeregisterAllContexts();

		lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
		parameters.SetMultiplicativeDepth(L);
		parameters.SetFirstModSize(firstMod);
		parameters.SetScalingModSize(scaleMod);
		parameters.SetBatchSize(8);
		parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
		parameters.SetRingDim(1 << logN);
		parameters.SetNumLargeDigits(dnum);
		parameters.SetScalingTechnique(scaleTech);
		parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
		parameters.SetPREMode(lbcrypto::INDCPA);

		lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
		try {
			cc = GenCryptoContext(parameters);
		} catch (const std::exception& e) {
			std::cout << "[bprec] logN=" << logN << " scaleMod=" << scaleMod << " context generation failed: " << e.what() << std::endl;
			continue;
		}
		cc->Enable(lbcrypto::PKE);
		cc->Enable(lbcrypto::KEYSWITCH);
		cc->Enable(lbcrypto::LEVELEDSHE);
		cc->Enable(lbcrypto::ADVANCEDSHE);
		cc->Enable(lbcrypto::FHE);

		const int numSlots = slotsArg > 0 ? slotsArg : static_cast<int>(cc->GetRingDimension()) / 2;

		FIDESlib::CKKS::Parameters fp{ .logN = logN, .L = L, .dnum = dnum, .primes = p64, .Sprimes = sp64 };
		FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
		FIDESlib::CKKS::Context GPUcc_		= FIDESlib::CKKS::GenCryptoContextGPU(fp.adaptTo(raw_param), devices);
		FIDESlib::CKKS::ContextData& GPUcc	= *GPUcc_;
		GPUcc.batch							= 128;

		const double logQ  = BPrecLogModulus(raw_param.moduli);
		const double logP  = BPrecLogModulus(raw_param.SPECIALmoduli);
		const double logQP = logQ + logP;

		auto keys = cc->KeyGen();
		cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0, true, false);
		cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
		FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

		// Only meaningful once EvalBootstrapSetup has run: it is the setup that
		// fits the correction factor to the ring degree and slot count. Reading
		// it off the RawParams captured before setup yields garbage.
		const uint32_t correctionFactor = cc->GetScheme()->m_FHE->GetCKKSBootCorrectionFactor();

		// A full slot vector, so the reported worst case really is the worst
		// case: precision is a per-slot property and the linear transforms mix
		// the slots, so eight probe values would understate it.
		std::mt19937 rng(20250806u + static_cast<unsigned>(scaleMod));
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		std::vector<double> x(numSlots);
		for (auto& v : x)
			v = dist(rng);

		lbcrypto::Plaintext ptxt		  = cc->MakeCKKSPackedPlaintext(x, 1, GPUcc.L - 1, nullptr, numSlots);
		auto c1							  = cc->Encrypt(keys.publicKey, ptxt);
		FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);

		double gpuWorst = -1.0, gpuRms = -1.0;
		int gpuLevel	= -1;
		std::string gpuNote;
		try {
			FIDESlib::CKKS::Ciphertext GPUct(GPUcc_, raw);
			Bootstrap(GPUct, numSlots, false);
			cudaDeviceSynchronize();
			gpuLevel = GPUct.getLevel();

			FIDESlib::CKKS::RawCipherText raw_res;
			GPUct.store(raw_res);
			auto result(c1);
			GetOpenFHECipherText(result, raw_res);

			lbcrypto::Plaintext result_pt;
			cc->Decrypt(keys.secretKey, result, &result_pt);
			result_pt->SetLength(numSlots);
			const BPrecError e = BPrecMeasure(x, result_pt->GetRealPackedValue());
			gpuWorst		   = e.worst;
			gpuRms			   = e.rms;
		} catch (const std::exception& e) {
			gpuNote = std::string(" gpuFailed=\"") + e.what() + "\"";
		}

		double cpuWorst = -1.0, cpuRms = -1.0;
		if (alsoCPU) {
			try {
				auto c2		 = cc->Encrypt(keys.publicKey, ptxt);
				auto cpuBoot = cc->EvalBootstrap(c2);
				lbcrypto::Plaintext cpu_pt;
				cc->Decrypt(keys.secretKey, cpuBoot, &cpu_pt);
				cpu_pt->SetLength(numSlots);
				const BPrecError e = BPrecMeasure(x, cpu_pt->GetRealPackedValue());
				cpuWorst		   = e.worst;
				cpuRms			   = e.rms;
			} catch (const std::exception& e) {
				std::cout << "[bprec]   cpu bootstrap failed: " << e.what() << std::endl;
			}
		}

		std::cout << "[bprec] logN=" << logN << " slots=" << numSlots << " scaleMod=" << scaleMod << " firstMod=" << firstMod << " deg=" << deg
				  << " correctionFactor=" << correctionFactor << " L=" << L << " limbs=" << raw_param.moduli.size()
				  << " Pprimes=" << raw_param.SPECIALmoduli.size() << " logQ=" << logQ << " logP=" << logP << " logQP=" << logQP
				  << " outLevel=" << gpuLevel << " gpuWorstErr=" << gpuWorst << " gpuBitsWorst=" << BPrecBits(gpuWorst)
				  << " gpuBitsRms=" << BPrecBits(gpuRms);
		if (alsoCPU)
			std::cout << " cpuWorstErr=" << cpuWorst << " cpuBitsWorst=" << BPrecBits(cpuWorst) << " cpuBitsRms=" << BPrecBits(cpuRms);
		std::cout << gpuNote << std::endl;

		cc->ClearEvalAutomorphismKeys();
		cc->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
		CKKS::DeregisterAllContexts();
		lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
	}
}

} // namespace FIDESlib::Testing