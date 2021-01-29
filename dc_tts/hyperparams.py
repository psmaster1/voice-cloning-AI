# /usr/bin/python2
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/css10
'''


class Hyperparams:
    '''Hyper parameters'''
    # lang = "kurztransfer"
    lang = "wolf"

    # pipeline
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.

    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)2048
    frame_shift = 0.0125  # seconds0.0125
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 80  # Number of Mel banks to generate 80
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20

    # Model
    r = 4  # Reduction factor. Do not change this.
    dropout_rate = 0.05
    e = 128  # == embedding
    d = 256  # == hidden units of Text2Mel
    c = 512  # == hidden units of SSRN
    attention_win_size = 3

    # data
    data = "data/private/{}/".format(lang)
    transcript = "data/private/{}/".format(lang)
    test_data = "../MOS/sents/{}.txt".format(lang)
    if lang == "de":
        vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜßàäéöü–'''
        max_N, max_T = 279, 435
    elif lang == "kurz":
        vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜßàäéöü–'''
        max_N, max_T = 279, 435
    elif lang == "kogler":
        vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜßàäéöü–'''
        max_N, max_T = 279, 435
    elif lang == "wolf":
        vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜßàäéöü–'''
        max_N, max_T = 279, 435

    # max_N = 180 # Maximum number of characters.
    # max_T = 210 # Maximum number of mel frames.

    # training scheme
    lr = 0.001  # Initial learning rate. 0.001
    logdir = "{}/logdir".format(lang)
    sampledir = '{}/samples'.format(lang)
    restoredir = 'de/logdir'.format(lang)
    B = 8  # batch size 8
    num_iterations = 2000000

    # select the trainable layers for transfer learning (i.e. remove the layers you want to fix during transfer learning)
    selected_tvars=[
         'SSRN/C_1/',
         'SSRN/HC_2/',
         'SSRN/HC_3/',
         'SSRN/D_4/',
         'SSRN/HC_5/',
         'SSRN/HC_6/',
         'SSRN/D_7/',
         'SSRN/HC_8/',
         'SSRN/HC_9/',
         'SSRN/C_10/',
         'SSRN/HC_11/',
         'SSRN/HC_12/',
         'SSRN/C_13/',
         'SSRN/C_14/',
         'SSRN/C_15/',
         'SSRN/C_16/',
         'Text2Mel/TextEnc/embed_1/',
         'Text2Mel/TextEnc/C_2/',
         'Text2Mel/TextEnc/C_3/',
         'Text2Mel/TextEnc/HC_4/',
         'Text2Mel/TextEnc/HC_5/',
         'Text2Mel/TextEnc/HC_6/',
         'Text2Mel/TextEnc/HC_7/',
         'Text2Mel/TextEnc/HC_8/',
         'Text2Mel/TextEnc/HC_9/',
         'Text2Mel/TextEnc/HC_10/',
         'Text2Mel/TextEnc/HC_11/',
         'Text2Mel/TextEnc/HC_12/',
         'Text2Mel/TextEnc/HC_13/',
         'Text2Mel/TextEnc/HC_14/',
         'Text2Mel/TextEnc/HC_15/',
         'Text2Mel/AudioEnc/C_1/',
         'Text2Mel/AudioEnc/C_2/',
         'Text2Mel/AudioEnc/C_3/',
         'Text2Mel/AudioEnc/HC_4/',
         'Text2Mel/AudioEnc/HC_5/',
         'Text2Mel/AudioEnc/HC_6/',
         'Text2Mel/AudioEnc/HC_7/',
         'Text2Mel/AudioEnc/HC_8/',
         'Text2Mel/AudioEnc/HC_9/',
         'Text2Mel/AudioEnc/HC_10/',
         'Text2Mel/AudioEnc/HC_11/',
         'Text2Mel/AudioEnc/HC_12/',
         'Text2Mel/AudioEnc/HC_13/',
         'Text2Mel/AudioDec/C_1/',
         'Text2Mel/AudioDec/HC_2/',
         'Text2Mel/AudioDec/HC_3/',
         'Text2Mel/AudioDec/HC_4/',
         'Text2Mel/AudioDec/HC_5/',
         'Text2Mel/AudioDec/HC_6/',
         'Text2Mel/AudioDec/HC_7/',
         'Text2Mel/AudioDec/C_8/',
         'Text2Mel/AudioDec/C_9/',
         'Text2Mel/AudioDec/C_10/',
         'Text2Mel/AudioDec/C_11/'
    ]

#
# selected_tvars = [
#                     'SSRN/C_1/',
#                     'SSRN/HC_2/',
#                     'SSRN/HC_3/',
#                     'SSRN/D_4/',
#                     'SSRN/HC_5/',
#                     'SSRN/HC_6/',
#                     'SSRN/D_7/',
#                     'SSRN/HC_8/',
#                     'SSRN/HC_9/',
#                     'SSRN/C_10/',
#                     'SSRN/HC_11/',
#                     'SSRN/HC_12/',
#                     'SSRN/C_13/',
#                     'SSRN/C_14/',
#                     'SSRN/C_15/',
#                     'SSRN/C_16/',
#                     'Text2Mel/TextEnc/embed_1/',
#                     'Text2Mel/TextEnc/C_2/',
#                     'Text2Mel/TextEnc/C_3/',
#                     'Text2Mel/TextEnc/HC_4/',
#                     'Text2Mel/TextEnc/HC_5/',
#                     'Text2Mel/TextEnc/HC_6/',
#                     'Text2Mel/TextEnc/HC_7/',
#                     'Text2Mel/TextEnc/HC_8/',
#                     'Text2Mel/TextEnc/HC_9/',
#                     'Text2Mel/TextEnc/HC_10/',
#                     'Text2Mel/TextEnc/HC_11/',
#                     'Text2Mel/TextEnc/HC_12/',
#                     'Text2Mel/TextEnc/HC_13/',
#                     'Text2Mel/TextEnc/HC_14/',
#                     'Text2Mel/TextEnc/HC_15/',
#                     'Text2Mel/AudioEnc/C_1/',
#                     'Text2Mel/AudioEnc/C_2/',
#                     'Text2Mel/AudioEnc/C_3/',
#                     'Text2Mel/AudioEnc/HC_4/',
#                     'Text2Mel/AudioEnc/HC_5/',
#                     'Text2Mel/AudioEnc/HC_6/',
#                     'Text2Mel/AudioEnc/HC_7/',
#                     'Text2Mel/AudioEnc/HC_8/',
#                     'Text2Mel/AudioEnc/HC_9/',
#                     'Text2Mel/AudioEnc/HC_10/',
#                     'Text2Mel/AudioEnc/HC_11/',
#                     'Text2Mel/AudioEnc/HC_12/',
#                     'Text2Mel/AudioEnc/HC_13/',
#                     'Text2Mel/AudioDec/C_1/',
#                     'Text2Mel/AudioDec/HC_2/',
#                     'Text2Mel/AudioDec/HC_3/',
#                     'Text2Mel/AudioDec/HC_4/',
#                     'Text2Mel/AudioDec/HC_5/',
#                     'Text2Mel/AudioDec/HC_6/',
#                     'Text2Mel/AudioDec/HC_7/',
#                     'Text2Mel/AudioDec/C_8/',
#                     'Text2Mel/AudioDec/C_9/',
#                     'Text2Mel/AudioDec/C_10/',
#                     'Text2Mel/AudioDec/C_11/'
#                     ]
#'SSRN/C_13/',
#'SSRN/C_14/',
#'SSRN/C_15/',
#'SSRN/C_16/',
#'Text2Mel/TextEnc/HC_11/',
#'Text2Mel/TextEnc/HC_12/',
#'Text2Mel/TextEnc/HC_13/',
#'Text2Mel/TextEnc/HC_14/',
#'Text2Mel/TextEnc/HC_15/',
#'Text2Mel/AudioEnc/HC_9/',
#'Text2Mel/AudioEnc/HC_10/',
#'Text2Mel/AudioEnc/HC_11/',
#'Text2Mel/AudioEnc/HC_12/',
#'Text2Mel/AudioEnc/HC_13/',
#'Text2Mel/AudioDec/HC_7/',
#'Text2Mel/AudioDec/C_8/',
#'Text2Mel/AudioDec/C_9/',
#'Text2Mel/AudioDec/C_10/',
#'Text2Mel/AudioDec/C_11/'

