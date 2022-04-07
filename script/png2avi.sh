#ffmpeg -f image2 -i %*.png out.avi
#ffmpeg  -i face_3/face_%05d.jpg -pix_fmt yuv420p -r 60 -y output.mp4
#ffmpeg -r 60  -i face_3/face_%05d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p output.mp4
#ffmpeg -r 1/5 -i face_3/face_%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4
ffmpeg -framerate 30 -pattern_type glob -i 'generated/*.jpg' -c:v libx264 -pix_fmt yuv420p -y v/homo.mp4

