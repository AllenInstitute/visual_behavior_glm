import os
import datetime
import argparse
import shutil
import json

default_ophys_sessions = [881236651, 880753403, 879897824, 878918807, 877946125, 876557230,
                          873542787, 945638542, 943299247, 942114744, 941300832, 940446507,
                          939521982, 938175546, 811310092, 809658968, 809393834, 808340530,
                          807393193, 806812738, 806203732, 799021448, 795866393, 791355484,
                          789220000, 788418567, 784219195, 783475910, 943297462, 942116236,
                          940447433, 939520170, 935554523, 934576432, 855083235, 854090022,
                          852326785, 850959102, 849597937, 848894137, 848264483, 929255698,
                          928142719, 923202821, 922168593, 920317769, 919432737, 918719819,
                          863527124, 858863712, 855753717, 855333357, 854524023, 853773937,
                          852794147, 907757569, 906303646, 903711544, 902777219, 901759468,
                          900802327, 898870044, 933604359, 931687751, 929423904, 928414538,
                          924644503, 922564930, 918889065, 918180797, 914728054, 918718249,
                          915306390, 914639324, 914161594, 913564409, 911719666, 908441202,
                          907753304, 906968227, 906299056, 904418381, 903621170, 865051382,
                          863981645, 860331718, 855762043, 855353286, 848815078, 848143026,
                          796018019, 792354406, 789746168, 787661032, 778015591, 776926613,
                          775749782, 775289198, 894975064, 894204946, 893413336, 892423647,
                          891465652, 890362408, 889235128, 931327989, 929688369, 929255931,
                          928146339, 918718550, 918116930, 914163299, 914634556, 912538650,
                          911724659, 911249905, 910739829, 910169030, 909229934, 889467038,
                          888171877, 887031077, 886367984, 885557130, 884613038, 882756028,
                          882386411, 880709154, 902617173, 898295293, 893937838, 892217026,
                          891193996, 889918231, 888940531, 812802821, 811234448, 809600993,
                          808107961, 807208045, 805989030, 824745199, 822388759, 820871900,
                          819949602, 817101568, 815485890, 853416014, 853141772, 851438454,
                          850959798, 849600749, 848891498, 848253761, 939919536, 935859569,
                          934908140, 933869822, 931972753, 927721370, 923704003, 848889656,
                          848264175, 846599329, 845219209, 843049997, 842583486, 959454463,
                          954961790, 954408500, 837363020, 836962545, 836322466, 835796136,
                          833812106, 833002992, 830272668, 908569379, 907991198, 907177554,
                          906521029, 904771513, 903813946, 902884228, 922743776, 918996859,
                          915587736, 914797752, 913834848, 911449165, 910971181, 893951816,
                          890086402, 889179793, 888190799, 884359468, 882351065, 797078933,
                          796202631, 796019065, 795217244, 794474159, 793857113, 792327341,
                          798007990, 796608428, 796236521, 796044280, 795625712, 794918442,
                          792619807, 836322359, 835795999, 833705055, 832881662, 830148632,
                          829521794, 825682242, 845842114, 845235947, 844469521, 843871999,
                          842623907, 842023261, 841682738, 941399977, 939524252, 938359348,
                          937420996, 933463604, 932664150, 931564063, 809261351, 808676476,
                          808092249, 807055274, 805269503, 803223329, 961180142, 960475393,
                          954958035, 954859502, 952434025, 949209988, 948090163, 942384133,
                          939524851, 949368499, 948252173, 946015345, 945124131, 940775208,
                          938898514, 937682841, 940448261, 939526443, 938140092, 937162622,
                          935559843, 933439847, 931326814, 929686773, 929255311, 933461266,
                          932662051, 928400396, 922511481, 853177377, 851740017, 850894918,
                          848983781, 848401585, 847758278, 846871218, 889944877, 889042121,
                          888009781, 886806800, 886130638, 884451806, 882674040, 882060185,
                          881094781, 952686746, 950031363, 949217880, 948042811, 947199653,
                          944888114, 942628468, 941676716, 939868908, 938454538, 842364341,
                          841778484, 841303580, 840490733, 839514418, 839208243, 885653926,
                          873247524, 872592724, 871526950, 870352564, 869117575, 927787876,
                          926774218, 925478114, 923705570, 921922878, 920695792, 919888953,
                          919041767, 917498735, 916650386, 960475921, 959458018, 958772311,
                          958105827, 957020350, 955775716, 954954402, 952430817, 951410079,
                          857040020, 855711263, 854060305, 853416532, 852794141, 850667270,
                          849304162, 797848297, 796032119, 795475729, 794706889, 794082263,
                          865854762, 865024413, 864458864, 863815473, 961226856, 960749169,
                          808387448, 807667681, 805643175, 804929369, 798818118, 961111597,
                          960593969, 959751299, 958931715, 957189583, 955991376, 954981981,
                          799195371, 796254988, 796047236, 875508749, 873720614, 870762788,
                          868688430, 867027875, 866197765]


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Make a new run JSON')
    # TODO Update this!
    default_model="/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/nick-allen/projects/ophys_glm/ophys_glm_toeplitz.py"
    default_manifest='/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/behavior_project_cache_20200127/manifest.json'

    parser.add_argument('--python-file-base', type=str, default=default_model,
                        metavar='/path/to/model.py',
                        help='model file to freeze')
    parser.add_argument('--manifest-path', type=str, default=default_manifest,
                        metavar='/path/to/manifest.json',
                        help='path to the allensdk data manifest')
    parser.add_argument('--run-name', type=str, default='.',
                        metavar='my_model_run',
                        help='name of the model run')
    args = parser.parse_args()
    
    # TODO Update paths
    output_dir_base = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'
    output_dir = output_dir_base + 'v_'+str(VERSION)
    os.mkdir(output_dir)
    model_freeze_dir = output_dir +'/frozen_model_files/'
    os.mkdir(model_freeze_dir)
    #model_freeze_dir = "/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/nick-allen/projects/ophys_glm/frozen_model_files"
    #job_dir_base = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_output/'
    #output_dir_base = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/ophys_glm'

    now = datetime.datetime.now().strftime('%Y%m%d')
    python_file_name = '{}_{}.py'.format(args.run_name, now)
    python_file_full_path = os.path.join(model_freeze_dir, python_file_name)
    shutil.copyfile(args.python_file_base, python_file_full_path)

    manifest_full_path = args.manifest_path

    #TODO: allow passing ophys sessions in as an argument somehow
    ophys_sessions = default_ophys_sessions

    run_params = {
        'manifest':manifest_full_path,
        'python_file':python_file_full_path,
        'job_dir':os.path.join(job_dir_base, args.run_name),
        'output_dir':os.path.join(output_dir_base, args.run_name),
        'regularization_lambda':70,
        'ophys_sessions':ophys_sessions
    }

    # Make job and output dirs
    if not os.path.exists(run_params['job_dir']):
        os.mkdir(run_params['job_dir'])
    if not os.path.exists(run_params['output_dir']):
        os.mkdir(run_params['output_dir'])

    run_params_full_path = os.path.join(run_params['output_dir'], 'run_params.json')

    with open(run_params_full_path, 'w') as json_file:
        json.dump(run_params, json_file, indent=4)

    print('Saved run params file: {}'.format(run_params_full_path))
