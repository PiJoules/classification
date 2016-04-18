% Code for reading an individual digit file from http://yann.lecun.com/exdb/mnist/
clear;
close all;
clc;

fid=fopen('data0','r');  % open the file corresponding to digit 0
[t1,N]=fread(fid,[28 28],'uchar');  % read in the first training example and store it in a 28x28 size matrix t1
[t2,N]=fread(fid,[28 28],'uchar');  % read the second example into t2

imshow(t1);
