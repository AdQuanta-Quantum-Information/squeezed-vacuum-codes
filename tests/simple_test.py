import numpy as np
from qutip import destroy, create, qeye

print('Testing Kraus operators')
dim = 5
gamma = 0.1
a = destroy(dim)
adag = create(dim)
n_op = adag @ a
I = qeye(dim)

# Current implementation
K0 = (I - 0.5 * gamma * n_op + gamma**2 * (0.125 * n_op @ n_op - 0.25 * n_op))
K1 = (gamma**0.5) * a
K2 = (gamma / 2.0**0.5) * (a @ a)

# Check completeness
sum_kraus = K0.dag() @ K0 + K1.dag() @ K1 + K2.dag() @ K2
max_diff = np.max(np.abs((sum_kraus - I).full()))
print(f'Max difference from identity: {max_diff:.2e}')
print(f'Expected O(gamma^3): {gamma**3:.2e}')

# Check K0 diagonal
k0_diag = np.real(np.diag(K0.full()))
print('K0 diagonal:', k0_diag)

# Expected from second-order expansion of exp(-gamma*n/2)
expected = 1 - gamma * np.arange(dim) / 2 + gamma**2 * np.arange(dim)**2 / 8
print('Expected exp(-gamma*n/2):', expected)
print('Difference:', np.abs(k0_diag - expected))
