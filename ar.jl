using BandedMatrices
using ToeplitzMatrices
using LinearAlgebra

function find_ρ(ϕ::Array{T}) where {T}
  p = length(ϕ)
  A = zeros(T, p+1,p+1)
  A[1,1] = 1;
  for i in 1:p
    for j in 1:p
        idx = abs(i-j)
        (idx <= p) && (A[i+1,idx+1] += ϕ[j])
    end
  end
  # Get the eigen vector associated with the largest eigen value.
  # This should correspond to the fix-point of ρ = A ρ
  r = eigvecs(A)[:,p+1];
  r /= r[1]
  # Get rid of the useless 1
  r[2:(p+1)]
end


function gen_ρ(n, ϕ::Array{T}) where {T}
  p = length(ϕ)
  ρ = zeros(T, n)
  ρ[1:p] = find_ρ(ϕ)
  for i in (p+1):n
    for j in 1:p
      ρ[i] += ϕ[j]*ρ[i-j]
    end
  end
  return ρ
end


function Γ_ar(n, ϕ::Array{T}) where {T}
  ρ = gen_ρ(n, ϕ)
  Toeplitz(ρ, ρ)
end


function F(n, ϕ::Array{T}, stationary=true) where {T}
  p = length(ϕ);
  # Actually, it will be a lower-band matrix with band-with equal to p
  M = BandedMatrix(zeros(T, n, n), (p,0));
  for i in 1:n
    M[i,i] = 1;
    for j in 1:p
      ((i-j) > 0) && (M[i,i-j] = -ϕ[j])
    end
  end
  if stationary
    E = Symmetric(BandedMatrix(M * Γ_ar(n, ϕ) * transpose(M), (p,p)));
    E = cholesky(E)
    # Uhh, very inefficient Gauss-algorithm and waste of mem/gc action.
    m = Matrix(M);
    M = BandedMatrix(E.L \ m, (p,0));
  end
  return M
end
