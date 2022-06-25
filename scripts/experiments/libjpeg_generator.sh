mkdir -p "/home/sean.man/datasets/ffhq_128/test_libjpeg/libjpeg"
mkdir -p "/home/sean.man/datasets/ffhq_128/test_libjpeg/libjpeg_float"
mkdir -p "/home/sean.man/datasets/ffhq_128/test_libjpeg/libjpeg_float_rgb"

for IMG in /home/sean.man/datasets/ffhq_128/test_libjpeg/*_orig.ppm
do
  base="$(basename -- $IMG)"
  suffix="_orig.ppm"
  file=${base%"$suffix"}
  cjpeg -quality 100 -sample 1x1,1x1,1x1 $IMG | djpeg -dct float > "/home/sean.man/datasets/ffhq_128/test_libjpeg/libjpeg/${file}.ppm"
  cjpeg -quality 100 -sample 1x1,1x1,1x1 -dct float $IMG | djpeg -dct float > "/home/sean.man/datasets/ffhq_128/test_libjpeg/libjpeg_float/${file}.ppm"
  cjpeg -quality 100 -sample 1x1,1x1,1x1 -dct float -rgb $IMG | djpeg -dct float > "/home/sean.man/datasets/ffhq_128/test_libjpeg/libjpeg_float_rgb/${file}.ppm"
done