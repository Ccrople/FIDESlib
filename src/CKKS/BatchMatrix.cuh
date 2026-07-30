//
// Batch matrix encoding, batch matrix encryption and batch CPMM.
//
// Implements Definition 1 (batch matrix encoding), Definition 2 (batch matrix
// encryption) and Algorithm 1 (batch CPMM) of Cheon, Kang and Lee, "Fast Batch
// Matrix Multiplication in Ciphertexts".
//
// R_N = Z[X]/(X^N + 1) is viewed as a rank-d module over the subring
// R_k = Z[Y]/(Y^k + 1) with Y = X^d and N = d * k. A matrix over R_k carries
// k/2 complex matrices at once, and matrix multiplication over R_k performs all
// k/2 products simultaneously.
//

#ifndef FIDESLIB_CKKS_BATCHMATRIX_CUH
#define FIDESLIB_CKKS_BATCHMATRIX_CUH

#include "CKKS/forwardDefs.cuh"
#include <complex>
#include <cstdint>
#include <vector>

namespace FIDESlib::CKKS {

/**
 * @brief Shape of a batch matrix operation.
 *
 * The row count @c d is the rank of R_N over R_k and therefore fixes both the
 * subring degree @c k = N/d and the batch size @c k/2. The column count is not
 * constrained by the subring and is carried separately: it is simply the number
 * of RLWE ciphertexts holding the matrix.
 */
struct BatchMatrixLayout {
	int N	  = 0; ///< Ring degree of the ambient ring R_N.
	int d	  = 0; ///< Matrix rows; rank of R_N as an R_k-module.
	int k	  = 0; ///< Subring degree, N/d.
	int batch = 0; ///< Number of complex matrices packed together, k/2.

	BatchMatrixLayout() = default;
	/** @throws std::invalid_argument if d does not divide N or is not a power of two. */
	BatchMatrixLayout(int N, int d);
};

/**
 * @brief Batch matrix encoding of Definition 1, evaluated on the host.
 *
 * Applies the length-k inverse DFT entrywise across a batch of complex matrices,
 * yielding a single matrix whose entries are elements of R_k. Because the
 * forward map m -> (m(zeta^{5^j}))_j is a ring homomorphism from R_k onto
 * C^{k/2}, multiplication in R_k realises the entire batch of complex matrix
 * products at once.
 *
 * This is a self-contained encoding: it is not OpenFHE's slot encoding, and
 * conversion between the two is not required for batch CPMM.
 */
class BatchMatrixEncoder {
  public:
	/** @param k Subring degree; must be a power of two and at least 2. */
	explicit BatchMatrixEncoder(int k);

	/** @brief Number of complex matrices carried per encoded matrix, k/2. */
	int slots() const {
		return k_ / 2;
	}
	/** @brief Subring degree k. */
	int degree() const {
		return k_;
	}

	/**
	 * @brief Encode a batch of complex matrices into one matrix over R_k.
	 *
	 * @param batch  k/2 matrices, each @p rows x @p cols in row-major order.
	 * @param rows   Row count of every matrix in the batch.
	 * @param cols   Column count of every matrix in the batch.
	 * @param Delta  CKKS scaling factor applied before rounding.
	 * @param out    Receives (rows * cols) * k coefficients; entry (i,j) occupies
	 *               the k-element run starting at (i * cols + j) * k.
	 */
	void Encode(const std::vector<std::vector<std::complex<double>>>& batch, int rows, int cols, double Delta, std::vector<int64_t>& out) const;

	/** @brief Inverse of Encode; recovers the k/2 complex matrices. */
	void Decode(const std::vector<int64_t>& in, int rows, int cols, double Delta, std::vector<std::vector<std::complex<double>>>& batch) const;

  private:
	int k_;
	/// pow_[j * k_ + t] = zeta^{5^j * t} with zeta = exp(i * pi / k_).
	std::vector<std::complex<double>> pow_;
};

/**
 * @brief A plaintext matrix over R_{q,k}, resident on the GPU in the length-k
 *        NTT domain.
 *
 * Holds one transformed copy per RNS limb, plus the Shoup precomputation used by
 * the modular GEMM. The plaintext operand is fixed across a CPMM, so paying for
 * Shoup factors once removes a Barrett reduction from every inner-loop product.
 */
class BatchMatrixPlaintext {
  public:
	/**
	 * @param cc     GPU context; a single device is required.
	 * @param layout Subring layout shared with the ciphertext operand.
	 * @param rows   Row count of the plaintext matrix (must match the ciphertext column count).
	 * @param cols   Column count of the plaintext matrix.
	 * @param level  RNS level the matrix is materialised at.
	 */
	BatchMatrixPlaintext(Context& cc, const BatchMatrixLayout& layout, int rows, int cols, int level);
	~BatchMatrixPlaintext();

	BatchMatrixPlaintext(BatchMatrixPlaintext&&) noexcept;
	BatchMatrixPlaintext(const BatchMatrixPlaintext&)			 = delete;
	BatchMatrixPlaintext& operator=(const BatchMatrixPlaintext&) = delete;

	/**
	 * @brief Upload encoded coefficients and transform them to the R_k NTT domain.
	 * @param coeffs Output of BatchMatrixEncoder::Encode for a @c rows x @c cols matrix.
	 */
	void Load(const std::vector<int64_t>& coeffs);

	int rows() const {
		return rows_;
	}
	int cols() const {
		return cols_;
	}
	int level() const {
		return level_;
	}
	/** @brief Scaling factor the coefficients were encoded with. */
	double NoiseFactor = 0;

	/// Transformed data, laid out as [limb][row][col][k] with the NTT index fastest.
	uint64_t* data() const {
		return dev_;
	}
	uint64_t* shoup() const {
		return dev_shoup_;
	}
	const std::vector<int>& primeIds() const {
		return primeids_;
	}
	const BatchMatrixLayout& layout() const {
		return layout_;
	}

  private:
	Context& cc_;
	BatchMatrixLayout layout_;
	int rows_;
	int cols_;
	int level_;
	int device_;
	std::vector<int> primeids_;
	uint64_t* dev_		 = nullptr;
	uint64_t* dev_shoup_ = nullptr;
};

/**
 * @brief Batch CPMM, Algorithm 1.
 *
 * Treats @p in as the columns of a matrix encryption (Definition 2): column j is
 * the RLWE ciphertext encrypting sum_i M[i][j] * X^i. Right-multiplies that
 * matrix encryption by @p U over R_{q,k} and rescales by the plaintext scaling
 * factor, so the result is a matrix encryption of the batch of products
 * {M_l * U_l}.
 *
 * The whole operation is two matrix products over R_{q,k}, one for each
 * ciphertext component, and involves no rotations or key switching.
 *
 * @param out Receives @c U.cols() ciphertexts; resized as needed.
 * @param in  @c U.rows() ciphertexts forming the columns of the matrix encryption.
 * @param U   Encoded plaintext matrix.
 * @param rescale Perform step 2 of Algorithm 1. Pass false to inspect the
 *                unscaled product, which is what the correctness tests compare.
 */
void BatchCPMM(std::vector<Ciphertext>& out, const std::vector<Ciphertext*>& in, const BatchMatrixPlaintext& U, bool rescale = true);

/**
 * @brief Build the R_N coefficient vectors of a matrix encryption's columns.
 *
 * Given the encoded matrix produced by BatchMatrixEncoder::Encode, returns the
 * @c cols coefficient vectors m_j = sum_{i<d} M[i][j] * X^i of Definition 2,
 * reduced modulo nothing: callers hand these to an RLWE encryption routine.
 *
 * @param coeffs Encoded matrix, (rows * cols) * k coefficients.
 * @param layout Subring layout; @c layout.d must equal @p rows.
 * @param out    Receives @p cols vectors of length @c layout.N.
 */
void BuildMatrixEncryptionCoefficients(const std::vector<int64_t>& coeffs, const BatchMatrixLayout& layout, int rows, int cols, std::vector<std::vector<int64_t>>& out);

/**
 * @brief Inverse of BuildMatrixEncryptionCoefficients.
 *
 * Recovers the encoded matrix from the coefficient vectors of the columns, so a
 * decrypted matrix encryption can be handed back to BatchMatrixEncoder::Decode.
 */
void SplitMatrixEncryptionCoefficients(const std::vector<std::vector<int64_t>>& in, const BatchMatrixLayout& layout, int rows, int cols, std::vector<int64_t>& coeffs);

} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_BATCHMATRIX_CUH
