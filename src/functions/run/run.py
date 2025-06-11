from modules import *
from functions.tests import *
from models import *
from collections import defaultdict

def run_fast(X,Y, num_permutations, kernel_name, ridge, band_factor, light = False):
    n = len(X); m = len(Y)
    fullsample = np.concatenate([X, Y])
    pairwise_dists = cdist(fullsample, fullsample, kernel_name)
    bandwidth = 2 * np.median(pairwise_dists[pairwise_dists > 0]) 
    obs_value = defaultdict()
    p_values = []
    permuted_values = defaultdict(list)
    for BF in band_factor:
        kernel_matrix  =  np.exp( - pairwise_dists / (bandwidth*BF)) 
        kxx, kyy, kxy = split_kernel(kernel_matrix, n, m)
        mX, mY = get_mvecs_X_Y(kxx, kxy, kyy)
        SX, SY = get_Smats_X_Y(kxx, kxy, kyy)
        SX = .5*(SX + SY); mX = .5*(mX + mY)  # symmetrize
        # eigvals_SX ,eigvecs_SX = SX_ED_from_kernel_via_svd(kernel_matrix[:,:n])
        eigvals_SX, eigvecs_SX=np.linalg.eigh(SX)
        C = get_Cmat(kernel_matrix)
        eigvals_C, eigvecs_C =np.linalg.eigh(C)
        obs_value['AggMMD', BF] = kxx.sum()/(n*(n-1)) +  kyy.sum()/(m*(m-1)) - 2* kxy.sum()/(n*m) -1e-17

        for r in ridge:
            inv_sqrtm_SX = inv_sqrtm_ED((eigvals_SX + r, eigvecs_SX))
            C_ED = (eigvals_C + r, eigvecs_C)
            innerHSmat_eigvals = np.linalg.eigvalsh(inv_sqrtm_SX @ (SY - SX) @ inv_sqrtm_SX)
            HS_term = np.sum(innerHSmat_eigvals**2)**.5
            innerHSmat_eigvals[innerHSmat_eigvals < -1 + 1e-5] = -1 +1e-5
            logdet2_term = np.abs(- np.sum(innerHSmat_eigvals) + np.sum(np.log(1 + innerHSmat_eigvals)) )
            # logdet2_term = HS_term
            CM_term = np.linalg.norm(inv_sqrtm_SX @ (mY - mX))
            obs_value['KLR', r, BF] =  CM_term + logdet2_term-1e-17
            obs_value['KLR-0', r, BF] = HS_term-1e-17
            obs_value['CM', r, BF] = CM_term-1e-17
            obs_value['SpecReg-MMD', r, BF] = np.linalg.norm(inv_sqrtm_ED(C_ED)@(mY-mX))-1e-17

        for _ in range(num_permutations):
            _kernel_matrix = kernel_matrix[np.ix_(perm := np.random.permutation(n + m), perm)]
            _kxx, _kyy, _kxy = split_kernel(_kernel_matrix, n, m)
            _mX, _mY = get_mvecs_X_Y(_kxx, _kxy, _kyy)
            _SX, _SY = get_Smats_X_Y(_kxx, _kxy, _kyy)
            _SX = .5*(_SX + _SY); _mX = .5*(_mX + _mY) # symmetrize
            # _eigvals_SX, _eigvecs_SX = SX_ED_from_kernel_via_svd(_kernel_matrix[:,:n])
            _eigvals_SX, _eigvecs_SX = np.linalg.eigh(_SX)
            _C = get_Cmat(_kernel_matrix)
            _eigvals_C, _eigvecs_C = np.linalg.eigh(_C)
            permuted_values['AggMMD', BF].append(_kxx.sum()/(n*(n-1)) +  _kyy.sum()/(m*(m-1)) - 2* _kxy.sum()/(n*m) > obs_value['AggMMD', BF])
            for r in  ridge:
                _SX_ED = (_eigvals_SX + r, _eigvecs_SX)
                _C_ED = (_eigvals_C + r, _eigvecs_C)
                _inv_sqrtm_SX = inv_sqrtm_ED(_SX_ED)
                _innerHSmat_eigvals = np.linalg.eigvalsh(_inv_sqrtm_SX @ (_SY - _SX) @ _inv_sqrtm_SX)
                _HSterm = np.sum(_innerHSmat_eigvals**2)**.5
                _innerHSmat_eigvals[_innerHSmat_eigvals < -1 + 1e-5] = -1 +1e-5
                _logdet2_term = np.abs(- np.sum(_innerHSmat_eigvals) + np.sum(np.log(1 + _innerHSmat_eigvals)) )
                # _logdet2_term = _HSterm
                _CM_term = np.linalg.norm(_inv_sqrtm_SX@(_mY-_mX))
                permuted_values['KLR-0', r, BF].append(_HSterm  > obs_value['KLR-0', r, BF])
                permuted_values['KLR', r, BF].append(_CM_term + _logdet2_term  > obs_value['KLR', r, BF])
                permuted_values['CM', r, BF].append( _CM_term > obs_value['CM', r, BF])
                permuted_values['SpecReg-MMD', r, BF].append(np.linalg.norm(inv_sqrtm_ED(_C_ED)@(_mY-_mX)) > obs_value['SpecReg-MMD', r, BF])

    p_values = defaultdict()
    p_values['AggMMD'] =  [ np.mean(permuted_values['AggMMD', BF]).astype(float) for BF in band_factor]
    
    for name  in [ 'KLR',  'KLR-0',  'CM', 'SpecReg-MMD']:
        p_values[name] = [np.mean(permuted_values[name, r, BF]).astype(float) for r in ridge for BF in band_factor]
    if not light:
        p_values['HT'] = [ HT_two_sample_test(X,Y)(num_permutations)[1] ]
        p_values['FR'] = [ FR_two_sample_test(X,Y)(num_permutations)[1] ]


    return {name: min(ps) < 0.05/len(ps) for name, ps in p_values.items()}

def run_fast_ModelIteration(n, d, _model_, model_params, num_permutations, kernel_name, ridge, band_factor, light = False):
    model = _model_(**model_params)(d)
    X = model.sample_X(n)
    Y = model.sample_Y(n)
    return run_fast(X, Y, num_permutations, kernel_name, ridge, band_factor)

def run_fast_parallel(n, d,  _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor , light = False):
    # iter_args = [(n, d, _model_,model_params, num_permutations, kernel_name, ridge, band_factor) for _ in range(N_iters)]
    # return Parallel(n_jobs=NUM_CORES)(delayed(run_fast_ModelIteration)(*args) for args in iter_args)
    iter_args = [(_model_(**model_params)(d).sample_X(n), _model_(**model_params)(d).sample_Y(n), num_permutations, kernel_name, ridge, band_factor, light) for _ in range(N_iters)]
    return Parallel(n_jobs=NUM_CORES)(delayed(run_fast)(*args) for args in iter_args)

