/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"

#include <vector>
#include <string>

//using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class SVMClass{
public:
	SVMClass();
	//~SVMClass();

	//int main_svmTrain(int argc, char **argv);
	int main_svmTrain(std::vector<std::string> input);	

	//int main_svmPredict(int argc, char **argv);
	int main_svmPredict(int predict_probability, 
						std::string test_file, 
						std::string model_file, 
						std::string output_file);

	int main_svmPredictFileless(int predict_probability, 
								std::vector<svm_node *> testData, 
								struct svm_model* model, 
								std::vector<double>& labels, 
								std::vector<std::vector<double>>& classProbability);

private:
	//--------train ----------------
	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob;		// set by read_problem
	//struct svm_model *model;
	struct svm_node *x_space;
	int cross_validation;
	int nr_fold;
	/*static*/ char *line;
	/*static*/ int max_line_len;

	//--------predict ----------------
	//static int (*info)(const char *fmt,...) = &printf;
	struct svm_node *x;
	int max_nr_attr;
	struct svm_model* model;
	int predict_probability;
	//static char *line = NULL;
	//static int max_line_len;

	void print_null_svmTrain(const char *s);
	int print_null_svmPredict(const char *s,...);
	void exit_with_help_svmTrain();
	void exit_with_help_svmPredict();
	void exit_input_error(int line_num);
	/*static*/ char* readline(FILE *input);

	void do_cross_validation();
	//void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
	void parse_command_line(std::vector<std::string> input, char *input_file_name, char *model_file_name);
	void read_problem(const char *filename);

	void predict(FILE *input, FILE *output);
};