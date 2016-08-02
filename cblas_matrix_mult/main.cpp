#include <iostream>

#include "matrix_multi.hpp"
#include "softmax.hpp"
#include "im2col.hpp"

#include "gtest/gtest.h"

//TEST(MULTITEST, TestA) {
//	for(int j = 0; j < 100; j++)
//	{
//	double *A,*B,*C,*C2;
//	int m,n,k;
//	
//	generate_data_prod(A,B,m,n,k);
//	
//	C = new double[m * n];
//	C2 = new double[m * n];
//	
//	prod(A,B,C,m,n,k);
//	prodEigen(A,B,C2,m,n,k);
//	for(int i = 0; i < m * n; i++)
//	{	
//		EXPECT_NEAR(C[i],C2[i],1e-6);
//	}

//	delete[] A;
//	delete[] B;
//	delete[] C;
//	delete[] C2;
//	}
//}

//TEST(MULTITEST, IM2COL) {
//	double im[]={1.68081682345 , -2.86493731019 , -0.358784224244 , 0.741236051526 , -0.9057275183 , 0.80420660888 , 0.275627536973 , 1.40278822599 , 0.912866037545 , 0.132805287383 , 1.42466605358 , 0.641477980884 , -1.01704577453 , 0.523097727945 , 1.18919870971 , 2.1465133573 , 0.72766865979 , -0.0472942071633 , -0.407887645619 , 0.0601970665634 , -0.537871905252 , 0.686150517411 , 0.20655779821 , 0.402185679393 , 0.684419142151 , -1.18544478558 , 0.564783759926 , -2.20064143063 , -0.0960236176444 , -1.20749486309 , -0.0657823884862 , -0.437233888681 , 0.496508064706 , 1.35099938255 , -1.1172067416 , 1.87383682727 , -1.76798344268 , -0.585826627619 , 0.4408188234 , -0.360925023585 , -0.821660008075 , 0.171729603867 , -0.254689845428 , 0.0451821587801 , 1.30752083015 , 0.536051464073 , 1.34836857546 , 0.648147102302 , -1.9362922724 , -0.362333314332 , -1.88724212464 , -1.46152814981 , -1.58633441136 , 0.435576917723 , -0.110472931401 , 0.615666024667 , -0.971206803207 , 0.896480946953 , 0.104964586037 , 0.0675315946177 , -0.419930122855 , 0.422468828184 , 0.894012742005 , -1.47039671371 , 1.80317218647 , -0.648253324547 , -0.410163600861 , 0.979167379662 , 0.18961427189 , -1.51960082318 , -1.11902104486 , 0.00607650312965 , -1.73220507678 , -0.694570781765 , -0.323466627844 , -0.517696006579 , -0.314151041471 , 1.29437059221 , 1.7328373763 , 0.0256518466138 , -0.902583023588 , -1.42820626088 , 0.90110795149 , 1.12388134487 , -0.481978399881 , -0.766792047129 , -0.912809742246 , 0.438818642184 , -1.14629468369 , 1.07417733464 , 0.617985106273 , 0.243538219528 , 0.420963350468 , 0.106013945361 , -0.642760700379 , 1.19821821808 , -0.612332361902 , -0.0283662505245 , -0.361873163543 , -0.541901336183};

//	double col_e[]={0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 1.68081682345 , -2.86493731019 , -0.358784224244 , 0.0 , 0.0 , 1.42466605358 , 0.641477980884 , -1.01704577453 , 0.0 , 0.0 , -0.537871905252 , 0.686150517411 , 0.20655779821 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 1.68081682345 , -2.86493731019 , -0.358784224244 , 0.741236051526 , -0.9057275183 , 1.42466605358 , 0.641477980884 , -1.01704577453 , 0.523097727945 , 1.18919870971 , -0.537871905252 , 0.686150517411 , 0.20655779821 , 0.402185679393 , 0.684419142151 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.358784224244 , 0.741236051526 , -0.9057275183 , 0.80420660888 , 0.275627536973 , -1.01704577453 , 0.523097727945 , 1.18919870971 , 2.1465133573 , 0.72766865979 , 0.20655779821 , 0.402185679393 , 0.684419142151 , -1.18544478558 , 0.564783759926 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.9057275183 , 0.80420660888 , 0.275627536973 , 1.40278822599 , 0.912866037545 , 1.18919870971 , 2.1465133573 , 0.72766865979 , -0.0472942071633 , -0.407887645619 , 0.684419142151 , -1.18544478558 , 0.564783759926 , -2.20064143063 , -0.0960236176444 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.275627536973 , 1.40278822599 , 0.912866037545 , 0.132805287383 , 0.0 , 0.72766865979 , -0.0472942071633 , -0.407887645619 , 0.0601970665634 , 0.0 , 0.564783759926 , -2.20064143063 , -0.0960236176444 , -1.20749486309 , 0.0 , 0.0 , 0.0 , 1.68081682345 , -2.86493731019 , -0.358784224244 , 0.0 , 0.0 , 1.42466605358 , 0.641477980884 , -1.01704577453 , 0.0 , 0.0 , -0.537871905252 , 0.686150517411 , 0.20655779821 , 0.0 , 0.0 , -0.0657823884862 , -0.437233888681 , 0.496508064706 , 0.0 , 0.0 , -0.821660008075 , 0.171729603867 , -0.254689845428 , 1.68081682345 , -2.86493731019 , -0.358784224244 , 0.741236051526 , -0.9057275183 , 1.42466605358 , 0.641477980884 , -1.01704577453 , 0.523097727945 , 1.18919870971 , -0.537871905252 , 0.686150517411 , 0.20655779821 , 0.402185679393 , 0.684419142151 , -0.0657823884862 , -0.437233888681 , 0.496508064706 , 1.35099938255 , -1.1172067416 , -0.821660008075 , 0.171729603867 , -0.254689845428 , 0.0451821587801 , 1.30752083015 , -0.358784224244 , 0.741236051526 , -0.9057275183 , 0.80420660888 , 0.275627536973 , -1.01704577453 , 0.523097727945 , 1.18919870971 , 2.1465133573 , 0.72766865979 , 0.20655779821 , 0.402185679393 , 0.684419142151 , -1.18544478558 , 0.564783759926 , 0.496508064706 , 1.35099938255 , -1.1172067416 , 1.87383682727 , -1.76798344268 , -0.254689845428 , 0.0451821587801 , 1.30752083015 , 0.536051464073 , 1.34836857546 , -0.9057275183 , 0.80420660888 , 0.275627536973 , 1.40278822599 , 0.912866037545 , 1.18919870971 , 2.1465133573 , 0.72766865979 , -0.0472942071633 , -0.407887645619 , 0.684419142151 , -1.18544478558 , 0.564783759926 , -2.20064143063 , -0.0960236176444 , -1.1172067416 , 1.87383682727 , -1.76798344268 , -0.585826627619 , 0.4408188234 , 1.30752083015 , 0.536051464073 , 1.34836857546 , 0.648147102302 , -1.9362922724 , 0.275627536973 , 1.40278822599 , 0.912866037545 , 0.132805287383 , 0.0 , 0.72766865979 , -0.0472942071633 , -0.407887645619 , 0.0601970665634 , 0.0 , 0.564783759926 , -2.20064143063 , -0.0960236176444 , -1.20749486309 , 0.0 , -1.76798344268 , -0.585826627619 , 0.4408188234 , -0.360925023585 , 0.0 , 1.34836857546 , 0.648147102302 , -1.9362922724 , -0.362333314332 , 0.0 , 0.0 , 0.0 , -0.537871905252 , 0.686150517411 , 0.20655779821 , 0.0 , 0.0 , -0.0657823884862 , -0.437233888681 , 0.496508064706 , 0.0 , 0.0 , -0.821660008075 , 0.171729603867 , -0.254689845428 , 0.0 , 0.0 , -1.88724212464 , -1.46152814981 , -1.58633441136 , 0.0 , 0.0 , -0.419930122855 , 0.422468828184 , 0.894012742005 , -0.537871905252 , 0.686150517411 , 0.20655779821 , 0.402185679393 , 0.684419142151 , -0.0657823884862 , -0.437233888681 , 0.496508064706 , 1.35099938255 , -1.1172067416 , -0.821660008075 , 0.171729603867 , -0.254689845428 , 0.0451821587801 , 1.30752083015 , -1.88724212464 , -1.46152814981 , -1.58633441136 , 0.435576917723 , -0.110472931401 , -0.419930122855 , 0.422468828184 , 0.894012742005 , -1.47039671371 , 1.80317218647 , 0.20655779821 , 0.402185679393 , 0.684419142151 , -1.18544478558 , 0.564783759926 , 0.496508064706 , 1.35099938255 , -1.1172067416 , 1.87383682727 , -1.76798344268 , -0.254689845428 , 0.0451821587801 , 1.30752083015 , 0.536051464073 , 1.34836857546 , -1.58633441136 , 0.435576917723 , -0.110472931401 , 0.615666024667 , -0.971206803207 , 0.894012742005 , -1.47039671371 , 1.80317218647 , -0.648253324547 , -0.410163600861 , 0.684419142151 , -1.18544478558 , 0.564783759926 , -2.20064143063 , -0.0960236176444 , -1.1172067416 , 1.87383682727 , -1.76798344268 , -0.585826627619 , 0.4408188234 , 1.30752083015 , 0.536051464073 , 1.34836857546 , 0.648147102302 , -1.9362922724 , -0.110472931401 , 0.615666024667 , -0.971206803207 , 0.896480946953 , 0.104964586037 , 1.80317218647 , -0.648253324547 , -0.410163600861 , 0.979167379662 , 0.18961427189 , 0.564783759926 , -2.20064143063 , -0.0960236176444 , -1.20749486309 , 0.0 , -1.76798344268 , -0.585826627619 , 0.4408188234 , -0.360925023585 , 0.0 , 1.34836857546 , 0.648147102302 , -1.9362922724 , -0.362333314332 , 0.0 , -0.971206803207 , 0.896480946953 , 0.104964586037 , 0.0675315946177 , 0.0 , -0.410163600861 , 0.979167379662 , 0.18961427189 , -1.51960082318 , 0.0 , 0.0 , 0.0 , -0.821660008075 , 0.171729603867 , -0.254689845428 , 0.0 , 0.0 , -1.88724212464 , -1.46152814981 , -1.58633441136 , 0.0 , 0.0 , -0.419930122855 , 0.422468828184 , 0.894012742005 , 0.0 , 0.0 , -1.11902104486 , 0.00607650312965 , -1.73220507678 , 0.0 , 0.0 , -0.902583023588 , -1.42820626088 , 0.90110795149 , -0.821660008075 , 0.171729603867 , -0.254689845428 , 0.0451821587801 , 1.30752083015 , -1.88724212464 , -1.46152814981 , -1.58633441136 , 0.435576917723 , -0.110472931401 , -0.419930122855 , 0.422468828184 , 0.894012742005 , -1.47039671371 , 1.80317218647 , -1.11902104486 , 0.00607650312965 , -1.73220507678 , -0.694570781765 , -0.323466627844 , -0.902583023588 , -1.42820626088 , 0.90110795149 , 1.12388134487 , -0.481978399881 , -0.254689845428 , 0.0451821587801 , 1.30752083015 , 0.536051464073 , 1.34836857546 , -1.58633441136 , 0.435576917723 , -0.110472931401 , 0.615666024667 , -0.971206803207 , 0.894012742005 , -1.47039671371 , 1.80317218647 , -0.648253324547 , -0.410163600861 , -1.73220507678 , -0.694570781765 , -0.323466627844 , -0.517696006579 , -0.314151041471 , 0.90110795149 , 1.12388134487 , -0.481978399881 , -0.766792047129 , -0.912809742246 , 1.30752083015 , 0.536051464073 , 1.34836857546 , 0.648147102302 , -1.9362922724 , -0.110472931401 , 0.615666024667 , -0.971206803207 , 0.896480946953 , 0.104964586037 , 1.80317218647 , -0.648253324547 , -0.410163600861 , 0.979167379662 , 0.18961427189 , -0.323466627844 , -0.517696006579 , -0.314151041471 , 1.29437059221 , 1.7328373763 , -0.481978399881 , -0.766792047129 , -0.912809742246 , 0.438818642184 , -1.14629468369 , 1.34836857546 , 0.648147102302 , -1.9362922724 , -0.362333314332 , 0.0 , -0.971206803207 , 0.896480946953 , 0.104964586037 , 0.0675315946177 , 0.0 , -0.410163600861 , 0.979167379662 , 0.18961427189 , -1.51960082318 , 0.0 , -0.314151041471 , 1.29437059221 , 1.7328373763 , 0.0256518466138 , 0.0 , -0.912809742246 , 0.438818642184 , -1.14629468369 , 1.07417733464 , 0.0 , 0.0 , 0.0 , -0.419930122855 , 0.422468828184 , 0.894012742005 , 0.0 , 0.0 , -1.11902104486 , 0.00607650312965 , -1.73220507678 , 0.0 , 0.0 , -0.902583023588 , -1.42820626088 , 0.90110795149 , 0.0 , 0.0 , 0.617985106273 , 0.243538219528 , 0.420963350468 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.419930122855 , 0.422468828184 , 0.894012742005 , -1.47039671371 , 1.80317218647 , -1.11902104486 , 0.00607650312965 , -1.73220507678 , -0.694570781765 , -0.323466627844 , -0.902583023588 , -1.42820626088 , 0.90110795149 , 1.12388134487 , -0.481978399881 , 0.617985106273 , 0.243538219528 , 0.420963350468 , 0.106013945361 , -0.642760700379 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.894012742005 , -1.47039671371 , 1.80317218647 , -0.648253324547 , -0.410163600861 , -1.73220507678 , -0.694570781765 , -0.323466627844 , -0.517696006579 , -0.314151041471 , 0.90110795149 , 1.12388134487 , -0.481978399881 , -0.766792047129 , -0.912809742246 , 0.420963350468 , 0.106013945361 , -0.642760700379 , 1.19821821808 , -0.612332361902 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 1.80317218647 , -0.648253324547 , -0.410163600861 , 0.979167379662 , 0.18961427189 , -0.323466627844 , -0.517696006579 , -0.314151041471 , 1.29437059221 , 1.7328373763 , -0.481978399881 , -0.766792047129 , -0.912809742246 , 0.438818642184 , -1.14629468369 , -0.642760700379 , 1.19821821808 , -0.612332361902 , -0.0283662505245 , -0.361873163543 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.410163600861 , 0.979167379662 , 0.18961427189 , -1.51960082318 , 0.0 , -0.314151041471 , 1.29437059221 , 1.7328373763 , 0.0256518466138 , 0.0 , -0.912809742246 , 0.438818642184 , -1.14629468369 , 1.07417733464 , 0.0 , -0.612332361902 , -0.0283662505245 , -0.361873163543 , -0.541901336183 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0};
//	double *col_p = new double[625];
//	im2col(im,col_p,1,1,10,10,5,5,2,2);
//	for(int i = 0; i < 625; i++)
//	{	
//		EXPECT_NEAR(col_p[i],col_e[i],1e-5);
//	}
//	delete[] col_p;
//}
//TEST(MULTITEST, COL2IM) {
//	double col[] = {0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.191479330517 , -0.560209772104 , 1.9815500995 , 0.0 , 0.0 , -1.80097743462 , 1.96981857806 , -0.135190625826 , 0.0 , 0.0 , 1.22961730301 , -1.67150131685 , 0.967046671828 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.191479330517 , -0.560209772104 , 1.9815500995 , -0.0664040013521 , -0.39896463817 , -1.80097743462 , 1.96981857806 , -0.135190625826 , 0.831061347133 , 0.856468512311 , 1.22961730301 , -1.67150131685 , 0.967046671828 , -0.0449131765274 , -0.795904510816 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 1.9815500995 , -0.0664040013521 , -0.39896463817 , 1.21103716206 , -0.593685025656 , -0.135190625826 , 0.831061347133 , 0.856468512311 , 0.144707734655 , -0.676856258013 , 0.967046671828 , -0.0449131765274 , -0.795904510816 , 1.13740909157 , 0.811628185231 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.39896463817 , 1.21103716206 , -0.593685025656 , -0.238593307859 , -0.68983153184 , 0.856468512311 , 0.144707734655 , -0.676856258013 , 0.670267725403 , -1.20564057052 , -0.795904510816 , 1.13740909157 , 0.811628185231 , 1.08129344211 , -0.83646132928 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.593685025656 , -0.238593307859 , -0.68983153184 , 0.428589623323 , 0.0 , -0.676856258013 , 0.670267725403 , -1.20564057052 , 0.968191351492 , 0.0 , 0.811628185231 , 1.08129344211 , -0.83646132928 , -1.58394821257 , 0.0 , 0.0 , 0.0 , -0.191479330517 , -0.560209772104 , 1.9815500995 , 0.0 , 0.0 , -1.80097743462 , 1.96981857806 , -0.135190625826 , 0.0 , 0.0 , 1.22961730301 , -1.67150131685 , 0.967046671828 , 0.0 , 0.0 , -0.464020727895 , -0.35072071535 , -0.185440948389 , 0.0 , 0.0 , -0.161659400255 , -0.864129110304 , 1.58295388265 , -0.191479330517 , -0.560209772104 , 1.9815500995 , -0.0664040013521 , -0.39896463817 , -1.80097743462 , 1.96981857806 , -0.135190625826 , 0.831061347133 , 0.856468512311 , 1.22961730301 , -1.67150131685 , 0.967046671828 , -0.0449131765274 , -0.795904510816 , -0.464020727895 , -0.35072071535 , -0.185440948389 , 0.823222895505 , 1.18186830392 , -0.161659400255 , -0.864129110304 , 1.58295388265 , 0.0646315090508 , -0.920808035265 , 1.9815500995 , -0.0664040013521 , -0.39896463817 , 1.21103716206 , -0.593685025656 , -0.135190625826 , 0.831061347133 , 0.856468512311 , 0.144707734655 , -0.676856258013 , 0.967046671828 , -0.0449131765274 , -0.795904510816 , 1.13740909157 , 0.811628185231 , -0.185440948389 , 0.823222895505 , 1.18186830392 , -1.2474965116 , -0.854166806597 , 1.58295388265 , 0.0646315090508 , -0.920808035265 , 1.47490313349 , -1.31118543634 , -0.39896463817 , 1.21103716206 , -0.593685025656 , -0.238593307859 , -0.68983153184 , 0.856468512311 , 0.144707734655 , -0.676856258013 , 0.670267725403 , -1.20564057052 , -0.795904510816 , 1.13740909157 , 0.811628185231 , 1.08129344211 , -0.83646132928 , 1.18186830392 , -1.2474965116 , -0.854166806597 , 0.494472014713 , -0.752825039019 , -0.920808035265 , 1.47490313349 , -1.31118543634 , 0.16800937477 , -0.56887855306 , -0.593685025656 , -0.238593307859 , -0.68983153184 , 0.428589623323 , 0.0 , -0.676856258013 , 0.670267725403 , -1.20564057052 , 0.968191351492 , 0.0 , 0.811628185231 , 1.08129344211 , -0.83646132928 , -1.58394821257 , 0.0 , -0.854166806597 , 0.494472014713 , -0.752825039019 , 0.516125902175 , 0.0 , -1.31118543634 , 0.16800937477 , -0.56887855306 , -0.03886840925 , 0.0 , 0.0 , 0.0 , 1.22961730301 , -1.67150131685 , 0.967046671828 , 0.0 , 0.0 , -0.464020727895 , -0.35072071535 , -0.185440948389 , 0.0 , 0.0 , -0.161659400255 , -0.864129110304 , 1.58295388265 , 0.0 , 0.0 , -0.597708687069 , -0.632859945672 , 0.15518544108 , 0.0 , 0.0 , 2.17946907346 , 0.312596433423 , -1.50595874553 , 1.22961730301 , -1.67150131685 , 0.967046671828 , -0.0449131765274 , -0.795904510816 , -0.464020727895 , -0.35072071535 , -0.185440948389 , 0.823222895505 , 1.18186830392 , -0.161659400255 , -0.864129110304 , 1.58295388265 , 0.0646315090508 , -0.920808035265 , -0.597708687069 , -0.632859945672 , 0.15518544108 , 0.893785020898 , -0.744230825286 , 2.17946907346 , 0.312596433423 , -1.50595874553 , -0.947575474248 , -0.911759231601 , 0.967046671828 , -0.0449131765274 , -0.795904510816 , 1.13740909157 , 0.811628185231 , -0.185440948389 , 0.823222895505 , 1.18186830392 , -1.2474965116 , -0.854166806597 , 1.58295388265 , 0.0646315090508 , -0.920808035265 , 1.47490313349 , -1.31118543634 , 0.15518544108 , 0.893785020898 , -0.744230825286 , 0.297475013139 , 1.73570089362 , -1.50595874553 , -0.947575474248 , -0.911759231601 , -0.864602111043 , -0.898011533527 , -0.795904510816 , 1.13740909157 , 0.811628185231 , 1.08129344211 , -0.83646132928 , 1.18186830392 , -1.2474965116 , -0.854166806597 , 0.494472014713 , -0.752825039019 , -0.920808035265 , 1.47490313349 , -1.31118543634 , 0.16800937477 , -0.56887855306 , -0.744230825286 , 0.297475013139 , 1.73570089362 , 0.526898286618 , 0.447848258253 , -0.911759231601 , -0.864602111043 , -0.898011533527 , -0.380332849859 , -0.244331589869 , 0.811628185231 , 1.08129344211 , -0.83646132928 , -1.58394821257 , 0.0 , -0.854166806597 , 0.494472014713 , -0.752825039019 , 0.516125902175 , 0.0 , -1.31118543634 , 0.16800937477 , -0.56887855306 , -0.03886840925 , 0.0 , 1.73570089362 , 0.526898286618 , 0.447848258253 , -0.0786023983012 , 0.0 , -0.898011533527 , -0.380332849859 , -0.244331589869 , -0.262719602272 , 0.0 , 0.0 , 0.0 , -0.161659400255 , -0.864129110304 , 1.58295388265 , 0.0 , 0.0 , -0.597708687069 , -0.632859945672 , 0.15518544108 , 0.0 , 0.0 , 2.17946907346 , 0.312596433423 , -1.50595874553 , 0.0 , 0.0 , -1.29361451675 , 0.541300852677 , 0.0319343497912 , 0.0 , 0.0 , -0.507633525834 , -0.277822908686 , 2.0396067632 , -0.161659400255 , -0.864129110304 , 1.58295388265 , 0.0646315090508 , -0.920808035265 , -0.597708687069 , -0.632859945672 , 0.15518544108 , 0.893785020898 , -0.744230825286 , 2.17946907346 , 0.312596433423 , -1.50595874553 , -0.947575474248 , -0.911759231601 , -1.29361451675 , 0.541300852677 , 0.0319343497912 , -0.0533087206677 , 0.138324417166 , -0.507633525834 , -0.277822908686 , 2.0396067632 , -1.57707214287 , -0.343646690503 , 1.58295388265 , 0.0646315090508 , -0.920808035265 , 1.47490313349 , -1.31118543634 , 0.15518544108 , 0.893785020898 , -0.744230825286 , 0.297475013139 , 1.73570089362 , -1.50595874553 , -0.947575474248 , -0.911759231601 , -0.864602111043 , -0.898011533527 , 0.0319343497912 , -0.0533087206677 , 0.138324417166 , -0.844267073843 , 0.387074135876 , 2.0396067632 , -1.57707214287 , -0.343646690503 , 0.678703638657 , 0.361996888953 , -0.920808035265 , 1.47490313349 , -1.31118543634 , 0.16800937477 , -0.56887855306 , -0.744230825286 , 0.297475013139 , 1.73570089362 , 0.526898286618 , 0.447848258253 , -0.911759231601 , -0.864602111043 , -0.898011533527 , -0.380332849859 , -0.244331589869 , 0.138324417166 , -0.844267073843 , 0.387074135876 , 0.233565052531 , 0.0357621700097 , -0.343646690503 , 0.678703638657 , 0.361996888953 , -1.1471625829 , 1.34925013961 , -1.31118543634 , 0.16800937477 , -0.56887855306 , -0.03886840925 , 0.0 , 1.73570089362 , 0.526898286618 , 0.447848258253 , -0.0786023983012 , 0.0 , -0.898011533527 , -0.380332849859 , -0.244331589869 , -0.262719602272 , 0.0 , 0.387074135876 , 0.233565052531 , 0.0357621700097 , 1.62463888922 , 0.0 , 0.361996888953 , -1.1471625829 , 1.34925013961 , 0.144560850444 , 0.0 , 0.0 , 0.0 , 2.17946907346 , 0.312596433423 , -1.50595874553 , 0.0 , 0.0 , -1.29361451675 , 0.541300852677 , 0.0319343497912 , 0.0 , 0.0 , -0.507633525834 , -0.277822908686 , 2.0396067632 , 0.0 , 0.0 , 0.224250765302 , 0.521099851135 , -0.271429708544 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 2.17946907346 , 0.312596433423 , -1.50595874553 , -0.947575474248 , -0.911759231601 , -1.29361451675 , 0.541300852677 , 0.0319343497912 , -0.0533087206677 , 0.138324417166 , -0.507633525834 , -0.277822908686 , 2.0396067632 , -1.57707214287 , -0.343646690503 , 0.224250765302 , 0.521099851135 , -0.271429708544 , -0.0151920501235 , -0.127390937451 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -1.50595874553 , -0.947575474248 , -0.911759231601 , -0.864602111043 , -0.898011533527 , 0.0319343497912 , -0.0533087206677 , 0.138324417166 , -0.844267073843 , 0.387074135876 , 2.0396067632 , -1.57707214287 , -0.343646690503 , 0.678703638657 , 0.361996888953 , -0.271429708544 , -0.0151920501235 , -0.127390937451 , -0.233860483095 , -1.09999646525 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.911759231601 , -0.864602111043 , -0.898011533527 , -0.380332849859 , -0.244331589869 , 0.138324417166 , -0.844267073843 , 0.387074135876 , 0.233565052531 , 0.0357621700097 , -0.343646690503 , 0.678703638657 , 0.361996888953 , -1.1471625829 , 1.34925013961 , -0.127390937451 , -0.233860483095 , -1.09999646525 , -0.568396235622 , 0.486737547016 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -0.898011533527 , -0.380332849859 , -0.244331589869 , -0.262719602272 , 0.0 , 0.387074135876 , 0.233565052531 , 0.0357621700097 , 1.62463888922 , 0.0 , 0.361996888953 , -1.1471625829 , 1.34925013961 , 0.144560850444 , 0.0 , -1.09999646525 , -0.568396235622 , 0.486737547016 , -0.540234945563 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0};
//	double im_e[] = {-0.765917322069 , -2.24083908841 , 11.889300597 , -0.265616005408 , -2.39378782902 , 4.84414864825 , -3.56211015393 , -0.954373231437 , -2.75932612736 , 0.857179246646 , -7.2039097385 , 7.87927431225 , -0.811143754955 , 3.32424538853 , 5.13881107386 , 0.578830938619 , -4.06113754808 , 2.68107090161 , -4.82256228208 , 1.93638270298 , 7.37770381803 , -10.0290079011 , 8.70342004646 , -0.269479059164 , -7.16314059735 , 6.82445454945 , 7.30465366708 , 6.48776065266 , -5.01876797568 , -4.7518446377 , -1.85608291158 , -1.4028828614 , -1.11264569034 , 3.29289158202 , 7.09120982354 , -4.98998604642 , -5.12500083958 , 1.97788805885 , -3.01130015608 , 1.03225180435 , -0.969956401532 , -5.18477466182 , 14.2465849439 , 0.387789054305 , -8.28727231738 , 8.84941880092 , -11.8006689271 , 1.00805624862 , -3.41327131836 , -0.11660522775 , -2.39083474828 , -2.53143978269 , 0.931112646477 , 3.57514008359 , -4.46538495172 , 1.18990005255 , 10.4142053617 , 2.10759314647 , 1.79139303301 , -0.157204796602 , 13.0768144408 , 1.87557860054 , -13.5536287097 , -5.68545284549 , -8.20583308441 , -5.18761266626 , -8.08210380174 , -2.28199709915 , -1.46598953921 , -0.788158806817 , -5.17445806702 , 2.16520341071 , 0.191606098747 , -0.213234882671 , 0.829946502999 , -3.37706829537 , 2.32244481526 , 0.934260210124 , 0.143048680039 , 3.24927777844 , -2.03053410333 , -1.11129163474 , 12.2376405792 , -6.30828857147 , -2.06188014302 , 2.71481455463 , 2.17198133372 , -4.58865033159 , 5.39700055846 , 0.289121700889 , 0.448501530603 , 1.04219970227 , -0.814289125633 , -0.030384100247 , -0.382172812352 , -0.467720966191 , -3.29998939575 , -1.13679247124 , 0.973475094032 , -0.540234945563};
//	double *im_p = new double[100];
//	memset(im_p,0,100*sizeof(double));
//	col2im(col,im_p,1,1,10,10,5,5,2,2);
//	for(int i = 0; i < 100; i++)
//	{
//		EXPECT_NEAR(im_p[i],im_e[i],1e-5);
//	}

//	
//	delete[] im_p;
//}
//TEST(MULTITEST, TestA) {
//	double *A,*B,*C,*C2;
//	int m,n,k;
//	
//	generate_data_transprod(A,B,m,n,k);
//	
//	C = new double[m * n];
//	C2 = new double[m * n];
//	
//	transProd(A,B,C,m,n,k);
//	transProdEigen(A,B,C2,m,n,k);
//	for(int i = 0; i < m * n; i++)
//	{	
//		EXPECT_NEAR(C[i],C2[i],1e-5);
//	}

//	delete[] A;
//	delete[] B;
//	delete[] C;
//	delete[] C2;
//}

//TEST(MULTITEST, TestA) {
//	double *A,*B,*C,*C2;
//	int m,n,k;
//	
//	generate_data_prodtrans(A,B,m,n,k);
//	
//	C = new double[m * n];
//	C2 = new double[m * n];
//	
//	prodTrans(A,B,C,m,n,k);
//	prodTransEigen(A,B,C2,m,n,k);
//	for(int i = 0; i < m * n; i++)
//	{	
//		EXPECT_NEAR(C[i],C2[i],1e-5);
//	}

//	delete[] A;
//	delete[] B;
//	delete[] C;
//	delete[] C2;
//}


//TEST(SOFTMAX,TestA) {
//	double x[] = {-0.41846516, -0.26175181,  0.48082043, -1.17560335, -0.56632416,
//			0.35940199, -0.13740556,  0.18600808,  1.14399182,  1.23932197};
//	double ye[] = {0.46090165,  0.53909835,  0.83975735,  0.16024265,  0.28379259,
//		    0.71620741,  0.41984404,  0.58015596,  0.47618549,  0.52381451};
//	double y[10] = {0};
//	softmax(x,y,5,1,1,2);
//	for(int i = 0; i < 10; i++)
//	{
//		EXPECT_NEAR(y[i],ye[i],1e-5);
//	}
//}

//TEST(ONEHOT,TestA)
//{
//	Real label[] = {0,1,2,0,1};
//	int num_classes = 3;
//	Real one_hot_e[] = {1,0,0,0,1,0,0,0,1,1,0,0,0,1,0};
//	Real one_hot_p[15] = {0};
//	onehot(label,one_hot_p,5,1,1,3);
//	for(int i = 0 ; i < 10; i++)
//	{
//		EXPECT_EQ(one_hot_p[i],one_hot_e[i]);
//	}
//}

//TEST(LOGSUNEXP,TestA)
//{
//	Real softmax[] = {88.48261261,  -569.94726562,  -639.72412109,   -21.12545776,   478.11630249,
//  -186.06669617,  -396.22387695,  1019.89892578,   423.5213623,   -387.02670288,
//   136.0594635,   -342.49804688,  -498.7484436,    -26.76722145,   560.17578125,
//     3.65353107,  -384.62591553,   754.26818848,   362.64297485,  -306.08300781,
//    84.22072601,  -215.98258972,  -663.27282715,  -331.40817261,   535.72344971,
//  -108.11855316,  -535.70849609,   733.73376465,   323.01593018,  -450.8789978,
//   194.0756073,   -342.93234253,  -447.59469604,   -46.96983337,   625.50592041,
//   -17.76305962,  -244.22669983,   847.0213623,    356.83370972,  -266.06970215,
//   -69.8292923,   -670.27453613,  -950.12744141,   237.00561523,   478.55371094,
//   146.15664673,  -326.22225952,   401.93527222,   150.81356812,  -104.47646332};
//	Real logsumexp_e[] = {1019.89892578,   754.26818848,   733.73376465,   847.0213623,    478.55371094};
//	Real logsumexp_p[5] = {0};
//	logsumexp(softmax,logsumexp_p,5,1,1,10);
//	for(int i = 0; i < 5; i++)
//	{
//		EXPECT_NEAR(logsumexp_p[i],logsumexp_p[i],1e-5);
//	}
//}
//int main()
//{
//double x[] = {-0.41846516, -0.26175181,  0.48082043, -1.17560335, -0.56632416,
//		0.35940199, -0.13740556,  0.18600808,  1.14399182,  1.23932197};
//double yp[] = {0.46090165,  0.53909835,  0.83975735,  0.16024265,  0.28379259,
//        0.71620741,  0.41984404,  0.58015596,  0.47618549,  0.52381451};
//double y[10] = {0};
//softmax(x,y,5,1,1,2);
//for(int i = 0; i < 10; i++)
//	cout << y[i] << endl;
//}

int main()
{
	double A[] = { 0.1398212 , -0.37395729, -0.82432563,  0.04155075,  0.2607413 ,
        1.82983096,  0.2402472 , -1.15238074, -0.79605108,  1.313406  ,
       -1.16944499, -0.19863522};
	pca(A,3,4);
}
