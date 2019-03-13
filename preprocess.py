'''
preprocess orig model to numpy version
a hacky conversion

discussion on pickle python 2/3 compatibility
https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3/41366785
https://stackoverflow.com/questions/46001958/typeerror-a-bytes-like-object-is-required-not-str-when-opening-python-2-pick/47814305#47814305
'''

import numpy as np
import sys

# -----------------------------------------------------------------------------

def convert_orig_to_py2():

    # run this script in python 2 with chumpy installed

    # import pickle
    import cPickle as pickle

    # src_path    = './models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    # output_path = './models/model_male_py2.pkl'
    src_path    = './models/basicmodel_f_lbs_10_207_0_v1.0.0.pkl'
    output_path = './models/model_female_py2.pkl'

    with open(src_path, 'rb') as f:
        src_data = pickle.load(f)
        print( "loaded model from %s" % src_path )

    model = {
        'J_regressor': src_data['J_regressor'],
        'weights': np.array(src_data['weights']),
        'posedirs': np.array(src_data['posedirs']),
        'v_template': np.array(src_data['v_template']),
        'shapedirs': np.array(src_data['shapedirs']),
        'f': np.array(src_data['f']),
        'kintree_table': src_data['kintree_table'],
        # 'joint_regressor': src_data['cocoplus_regressor']
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print( "saved at %s" % output_path )

    import ipdb; ipdb.set_trace()

# -----------------------------------------------------------------------------

def convert_py2_to_py3():

    # run in python 3

    # output_path  = './models/model_male_py2.pkl'
    # convert_path = './models/model_male_py3.pkl'
    output_path  = './models/model_female_py2.pkl'
    convert_path = './models/model_female_py3.pkl'

    with open( output_path, 'rb' ) as fp:
        model = np.load( fp, encoding='latin1' )

    import pickle
    with open( convert_path, 'wb' ) as fp:
        pickle.dump( model, fp )

    import ipdb; ipdb.set_trace()

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # convert_orig_to_py2()

    convert_py2_to_py3()