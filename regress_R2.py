import warnings
from scipy.stats import linregress

def regress_R2(OTFshifted_OTFoverlap_ang, DO1overlap6_ang, DO0overlap6_ang, Progressbar):
    # regress_R2(OTFshifted_OTFoverlap_ang(:,:,ii), DO1overlap6_ang(:,:,ii), DO0overlap6_ang(:,:,ii), Progressbar); 
    # check DO1overlap6_angles <-(OTFshifted_OTFoverlap_ang, DO1overlap6_ang)  
    warnings.filterwarnings('ignore', '.*NoConst*')
    R2_a_ang = OTFshifted_OTFoverlap_ang
    R2_b_ang = R2_a_ang.reshape(-1, 1)
    R2_b_ang[R2_b_ang != 0] = 1
    
    R2_c = DO1overlap6_ang
    R2_d = R2_c.reshape(-1, 1)
    R2_b_ang[:, 1] = R2_d
    
    R2_c = DO0overlap6_ang
    R2_d = R2_c.reshape(-1, 1)
    R2_b_ang[:, 2] = R2_d
    
    R2_b_ang = R2_b_ang[(R2_b_ang[:, 0] != 0), :]  # mask with OTF overlap
    R2_y = R2_b_ang[:, 1]  # DO1 overlap
    R2_x = R2_b_ang[:, 2]  # DO0 overlap
    
    slope, intercept, r_value, p_value, std_err = linregress(R2_x, R2_y)
    R2_angles = r_value ** 2
    dbgRegress = R2_angles
    
    return dbgRegress