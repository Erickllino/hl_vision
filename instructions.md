Nessa branch, eu quero fazer um programa que irá receber uma rosbag com a entrada da visao e passa pelo hsl-player para preprocessar essa visao e passar pelo modelo de detecção e segmentação.

Esse repo ira ficar dentro do robo e vai rodar o hsl-player de forma seemless, ou seja o hsl-player nao tem ideia que ele não está vendo a camera de verdade. 

O codigo ficara no formato

src/...
docs/
data/

A pasta data tem a rosbag.


Quero poder acessar o robo via ssh (ssh booster@192.168.0.102 -X) e ver uma interface onde eu consigo ver a imagem RAW com decteções e segmentações, a imagem preprocessada e tambem mexer nos parametros do preprocessamento em realtime