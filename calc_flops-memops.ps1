$data_path='D:\data\fixed_rate\'

.\x64\Release\RandomDataGenerator.exe -t 0 -n 1000 -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 0 -n 3000 -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 0 -n 5000 -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 0 -n 100000 -p $data_path

.\x64\Release\RandomDataGenerator.exe -t 1 -n 100000 -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 1 -n 100000 --isHeightNormalDist -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 1 -n 100000 --isWidthNormalDist -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 1 -n 100000 -c -e -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 2 -n 100000 -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 3 -n 100000 -p $data_path

.\x64\Release\RandomDataGenerator.exe -t 4 -n 100000 -s 1 -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 4 -n 100000 -s 1 --inverseSkewed -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 4 -n 100000 -s 5 -p $data_path
.\x64\Release\RandomDataGenerator.exe -t 4 -n 100000 -s 5 --inverseSkewed -p $data_path

futhark opencl ./futhark/flops-memops.fut -o ./futhark/flops-memops.exe 

echo $data_path'0_UNIFORM_1000.in'
cat $data_path'0_UNIFORM_1000.in' | ./futhark/flops-memops.exe
echo $data_path'0_UNIFORM_3000.in'
cat $data_path'0_UNIFORM_3000.in' | ./futhark/flops-memops.exe
echo $data_path'0_UNIFORM_5000.in'
cat $data_path'0_UNIFORM_5000.in' | ./futhark/flops-memops.exe
echo $data_path'0_UNIFORM_100000.in'
cat $data_path'0_UNIFORM_100000.in' | ./futhark/flops-memops.exe
echo $data_path'1_RAND_100000.in'
cat $data_path'1_RAND_100000.in' | ./futhark/flops-memops.exe
echo $data_path'1_RAND_NORMH_100000.in'
cat $data_path'1_RAND_NORMH_100000.in' | ./futhark/flops-memops.exe
echo $data_path'1_RAND_NORMW_100000.in'
cat $data_path'1_RAND_NORMW_100000.in' | ./futhark/flops-memops.exe
echo $data_path'4_SKEWED_1_100000.in'
cat $data_path'4_SKEWED_1_100000.in' | ./futhark/flops-memops.exe
echo $data_path'4_SKEWED_INV_1_100000.in'
cat $data_path'4_SKEWED_INV_1_100000.in' | ./futhark/flops-memops.exe