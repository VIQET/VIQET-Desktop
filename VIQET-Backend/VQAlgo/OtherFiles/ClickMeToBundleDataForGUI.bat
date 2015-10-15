mkdir VIQET_backend
mkdir VIQET_backend\GaborFiltersInText

copy MOSModelData_forAll.txt 					VIQET_backend
copy flatRegionModelData.txt 					VIQET_backend
copy FlatRegionClassificationSVM.RData 				VIQET_backend
copy SVM_VANILLADOT_MODEL_BASED_ON_PILOT1_PILOT2.RData 		VIQET_backend
copy SVM_FOOD_VANILLADOT_MODEL.RData 				VIQET_backend
copy SVM_LANDSCAPE_VANILLADOT_MODEL.RData 			VIQET_backend
copy SVM_NIGHT_VANILLADOT_MODEL.RData 				VIQET_backend
copy SVM_WALL_VANILLADOT_MODEL.RData 				VIQET_backend
copy SvmRScript.txt 						VIQET_backend
copy Rscript.txt 						VIQET_backend
copy RFood.txt 							VIQET_backend
copy RLandscape.txt 						VIQET_backend
copy RNight.txt 						VIQET_backend
copy RWall.txt 							VIQET_backend
copy VQAlgo.dll 						VIQET_backend
copy VQHelper.dll 						VIQET_backend
xcopy GaborFiltersInText 					VIQET_backend\GaborFiltersInText	/S
