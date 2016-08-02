# CMake generated Testfile for 
# Source directory: /home/zak/Code/cblas_matrix_mult
# Build directory: /home/zak/Code/cblas_matrix_mult
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test "/home/zak/Code/cblas_matrix_mult/runUnitTests")
subdirs(gtest)
