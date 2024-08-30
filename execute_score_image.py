from score_image import score_image
import asseti
import argparse
import os

call_path = os.getcwd()

# model can be over-written by a choice from the command line:
default_model     = 'm2,m4,m7'
default_outpath   = f'{call_path}/scored'

parser = argparse.ArgumentParser(prog='score_image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-i', '--image',  type=str, default=None, help='path to the ortho image to be scored')
parser.add_argument('-c', '--check',  type=int, default=0, help='check inputs and then quit')
parser.add_argument('-v', '--verbose',  type=int, default=1, help='whether to be quiet or not')
parser.add_argument('-d', '--device', type=str, default='cpu', help='select device from "cpu" / "0" (first gpu) / "0,1,2,3" (list of GPUs) or "mps" (apple silicon only) ')
parser.add_argument('-m', '--model',  type=str, default=default_model, help='model to use for scoring')
parser.add_argument('-o', '--outpath',  type=str, default=default_outpath, help='path to parent-dir to store outputs')
parser.add_argument('-b', '--batchsize',  type=int, default=25, help='number of image tiles to score at once')

parser.add_argument('-j', '--jsonoptions',  type=str, default='none', help='optional - JSON file to provide certain options')

parser.add_argument('-s', '--oversample',  type=int, default=90, help='percentage to oversample for tile overlapping')

parser.add_argument('--sx',  type=float, default=0.0, help='value used for scaleX (not used)')
parser.add_argument('--sy',  type=float, default=0.0, help='value used for scaleY (not used)')
parser.add_argument('--sz',  type=float, default=0.0, help='value used for scaleZ (not used)')

parser.add_argument('--rx',  type=float, default=0.0, help='value used for rotateX')
parser.add_argument('--ry',  type=float, default=0.0, help='value used for rotateY (not used)')
parser.add_argument('--rz',  type=float, default=0.0, help='value used for rotateZ (not used)')

parser.add_argument('--tx',  type=float, default=0.0, help='value used for translationX')
parser.add_argument('--ty',  type=float, default=0.0, help='value used for translationY')
parser.add_argument('--tz',  type=float, default=0.0, help='value used for translationZ (not used)')

args = parser.parse_args()

def verbosity(str):
    if args.verbose == 1:
        print(str)

#verbosity(args)

# ---------------------------------------
# Main dictionary for use with the pipeline:
main_menu = {}
# ---------------------------------------
# create score_options:
score_options = {}
score_options['image_path']   = args.image
score_options['device']       = args.device
score_options['model']        = args.model
score_options['modelpath']    = [args.model]
score_options['output_path']  = args.outpath
score_options['rotateX']      = args.rx
score_options['rotateY']      = args.ry
score_options['rotateZ']      = args.rz
score_options['scaleX']       = args.sx
score_options['scaleY']       = args.sy
score_options['scaleZ']       = args.sz
score_options['translationX'] = args.tx
score_options['translationY'] = args.ty
score_options['translationZ'] = args.tz
score_options['score_batch_size'] = args.batchsize
score_options['oversample']   = args.oversample
score_options['check']        = args.check

# ---------------------------------------

if args.jsonoptions != "none":
    js_opts = asseti.load_json(args.jsonoptions)
    for key in score_options.keys():
        if key in js_opts:
            score_options[key] = js_opts[key]


# ---------------------------------------
main_menu['score_options'] = score_options
# ---------------------------------------


score_image(main_menu)


# ---------------------------------------
