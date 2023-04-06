import warnings
from scipy.stats import linregress
import numpy as np

def regress_R2(OTFshifted_OTFoverlap_ang, DO1overlap6_ang, DO0overlap6_ang, Progressbar = False):
    # regress_R2(OTFshifted_OTFoverlap_ang(:,:,ii), DO1overlap6_ang(:,:,ii), DO0overlap6_ang(:,:,ii), Progressbar); 
    # check DO1overlap6_angles <-(OTFshifted_OTFoverlap_ang, DO1overlap6_ang)  
    warnings.filterwarnings('ignore', '.*NoConst*')
    R2_a_ang = OTFshifted_OTFoverlap_ang
    R2_b_ang = np.zeros((OTFshifted_OTFoverlap_ang.size, 3))
    R2_b_ang[:, 0] = np.ravel(R2_a_ang, order='F').reshape(-1)
    R2_b_ang[R2_b_ang != 0] = 1
    
    R2_c = DO1overlap6_ang
    R2_d = np.ravel(R2_c, order='F').reshape(-1)
    R2_b_ang[:, 1] = R2_d
    
    R2_c = DO0overlap6_ang
    R2_d = np.ravel(R2_c, order='F').reshape(-1)
    R2_b_ang[:, 2] = R2_d
    
    R2_b_ang = R2_b_ang[(R2_b_ang[:, 0] != 0), :]  # mask with OTF overlap
    R2_y = R2_b_ang[:, 1]  # DO1 overlap
    R2_x = R2_b_ang[:, 2]  # DO0 overlap
    
    slope, intercept, r_value, p_value, std_err = linregress(R2_x, R2_y)
    R2_angles = r_value ** 2
    dbgRegress = R2_angles
    
    return dbgRegress