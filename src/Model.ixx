#include <algorithm>
#include <iostream>
#include "json.hpp"
#include <fstream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>

import Flatten;
import Layer;
import Input;
import Dense;
import Conv;
import Data;

export module Model;


export class Model {
private:
	std::vector<Layer*> layers;
	int lastIndex = -1;
	Data* data;

	int batchSize = -1;

	void workerPredict(int stepNum, int targetIndex) {
		this->layers[0]->forward(this->data->getImgs()[targetIndex], stepNum);
		for (size_t x = 1; x < this->layers.size(); x++) {
			this->layers[x]->forward(stepNum);
		}
	}
	void workerValidate(int stepNum, int targetIndex) {
		// forward
		this->layers[0]->forward(this->data->getTestX()[targetIndex], stepNum);
		for (size_t x = 1; x < this->layers.size(); x++) {
			this->layers[x]->forward(stepNum);
		}
	}
	void workerTrain(int stepNum, int targetIndex) {
		// forward
		this->layers[0]->forward(this->data->getTrainX()[targetIndex], stepNum);
		for (size_t x = 1; x < this->layers.size(); x++) {
			this->layers[x]->forward(stepNum);
		}

		// backward
		// calc dB
		float* target = this->data->getTrainY()[targetIndex];
		float* predict = this->layers[this->lastIndex]->getOutputArray1D()[stepNum];
		float* dB = this->layers[lastIndex]->getdB1D()[stepNum];
		int dBX = this->layers[this->lastIndex]->getOutputX();
		for (int x = 0; x < dBX; x++) {
			dB[x] = (2.0f / (float)dBX) * (predict[x] - target[x]) * (predict[x] * (1.0f - predict[x]));
		}

		for (size_t x = this->lastIndex; x > 1; x--) {
			this->layers[x]->calcPrevdB(stepNum);
		}

		// calc dW
		for (size_t x = this->lastIndex; x > 1; x--) {
			this->layers[x]->calcdW(stepNum);
		}

		// progress bar xd
		//std::cout << '.';
	}
	std::vector<std::thread> threads;

	float validate() {
		int batchesNum = this->data->getTestNum() / this->batchSize;
		float ret = 0.0f;
		float** predict;
		float max;
		int maxIndex;
		float maxTest;
		int maxIndexTest;
		int labelsNum = this->data->getLabelsNum();
		float** testY = this->data->getTestY();

		for (int batchNum = 0; batchNum < batchesNum; batchNum++) {
			//std::cout << "batch " << batchNum << " start" << std::endl;

			// uruchom wszystkie watki
			for (int x = batchNum * this->batchSize; x < (batchNum + 1) * this->batchSize; x++) {
				//std::cout << x << ' ' << x - batchNum * this->batchSize << std::endl;
				this->threads.push_back(std::thread(&Model::workerValidate, this, x - batchNum * batchSize, x));
			}
			// poczekaj na wszystkie watki
			for (int x = 0; x < this->batchSize; x++) {
				this->threads[x].join();
			}
			this->threads.clear();

			predict = this->layers[this->lastIndex]->getOutputArray1D();
			for (int bs = 0; bs < this->batchSize; bs++) {
				max = predict[bs][0];
				maxIndex = 0;
				for (int x = 0; x < labelsNum; x++) {
					if (predict[bs][x] > max) {
						max = predict[bs][x];
						maxIndex = x;
					}
				}
				maxTest = testY[batchNum * batchSize + bs][0];
				maxIndexTest = 0;
				for (int x = 1; x < labelsNum; x++) {
					if (testY[batchNum * batchSize + bs][x] > maxTest) {
						maxTest = testY[batchNum * batchSize + bs][x];
						maxIndexTest = x;
					}
				}
				if (maxIndex == maxIndexTest) {
					ret++;
				}
			}

			//std::cout << std::endl;
		}

		return ret / (float)this->data->getTestNum() * 100.0f;
		//return ret;
	}

	bool saveWeight(std::string weightsPath) {
		nlohmann::json list = nlohmann::json::array();
		for (int x = 1; x <= this->lastIndex; x++) {
			nlohmann::json j = layers[x]->getWeight();
			if (!j.is_null()) {
				list.push_back(j[0]);
				list.push_back(j[1]);
			}
		}

		std::ofstream file(weightsPath);
		if (file.is_open()) {
			file << list;
			file.close();
			return true;
		}
		else {
			std::cout << "Nie mozna otworzyc pliku z wagami" << std::endl;
			return false;
		}
	}
public:
	Model() {
	}
	~Model() {
		for (int x = 0; x <= this->lastIndex; x++) {
			delete this->layers[x];
		}
	}
	void add(Layer* layer) {
		this->layers.push_back(layer);
		this->lastIndex++;
	}
	bool compile(float learningRate, int batchSize) {
		this->batchSize = batchSize;
		this->layers[0]->compile(batchSize);
		for (size_t x = 1; x <= this->lastIndex; x++) {
			if (this->layers[x]->compile(this->layers[x - 1], learningRate, batchSize) == false) {
				std::cout << "Niepoprawe wymiary warstw" << std::endl;
				return false;
			}
		}
		if (this->layers[this->lastIndex]->getOutputDimsCount() == 1) {
			return true;
		}
		else {
			std::cout << "Niepoprawe wymiary warstw" << std::endl;
			return false;
		}
	}

	bool train(int epochs, Data* data, std::string weightsPath) {
		this->data = data;
		int batchesNum = this->data->getTrainNum() / this->batchSize;

		std::cout << "Poczatkowa trafnosc: " << this->validate() << "%" << std::endl;

		std::vector<int> index(this->data->getTrainNum());
		std::generate(index.begin(), index.end(), [n = 0]() mutable {return n++; });
		std::random_device rd;
		std::mt19937 g(rd());

		for (int e = 0; e < epochs; e++) {
			auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
			std::cout << "Epoka: " << e;
			std::shuffle(index.begin(), index.end(), g);

			for (int batchNum = 0; batchNum < batchesNum; batchNum++) {
				for (int x = batchNum * this->batchSize; x < (batchNum + 1) * this->batchSize; x++) {
					this->threads.push_back(std::thread(&Model::workerTrain, this, x - batchNum * this->batchSize, index[x]));
				}
				for (int x = 0; x < this->batchSize; x++) {
					this->threads[x].join();
				}
				this->threads.clear();
				for (size_t x = this->lastIndex; x > 1; x--) {
					this->layers[x]->updateWeights(this->batchSize);
				}
				for (size_t x = this->lastIndex; x > 1; x--) {
					this->layers[x]->setdWToZero();
				}
			}
			auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
			std::cout << ", czas: " << end - start << ", trafnosc: " << this->validate() << "%" << std::endl;

			if (!this->saveWeight(weightsPath)) {
				std::cout << "Niepoprawna sciezka do wag" << std::endl;
				return false;
			}
		}
		return true;
	}
	std::string eval(Data* data) {
		this->data = data;
		int batchesNum = this->data->getImgsNum() / this->batchSize;
		float** predict;
		float max;
		int maxIndex;
		int labelsNum = this->data->getLabelsNum();
		std::vector<std::string> labels = this->data->getLabels();
		std::vector<std::string> imgsPath = this->data->getImgsPath();
		std::string ret;

		for (int batchNum = 0; batchNum < batchesNum; batchNum++) {
			for (int bs = 0; bs < this->batchSize; bs++) {
				this->threads.push_back(std::thread(&Model::workerPredict, this, bs, batchNum * this->batchSize + bs));
			}
			for (int bs = 0; bs < this->batchSize; bs++) {
				this->threads[bs].join();
			}
			this->threads.clear();

			predict = this->layers[this->lastIndex]->getOutputArray1D();
			for (int bs = 0; bs < this->batchSize; bs++) {
				max = predict[bs][0];
				maxIndex = 0;
				for (int x = 0; x < labelsNum; x++) {
					if (predict[bs][x] > max) {
						max = predict[bs][x];
						maxIndex = x;
					}
				}
				ret += "Plik: " + imgsPath[batchNum * this->batchSize + bs] + "\nPredykja: " + labels[maxIndex] + '\n';
				for (int x = 0; x < labelsNum; x++) {
					ret += labels[x] + ": " + std::to_string(predict[bs][x] * 100.0f) + "%\n";
				}
			}
		}

		for (int x = 0; x < this->data->getImgsNum() % this->batchSize; x++) {
			this->threads.push_back(std::thread(&Model::workerPredict, this, x, batchesNum * this->batchSize + x));
		}
		for (int x = 0; x < this->data->getImgsNum() % this->batchSize; x++) {
			this->threads[x].join();
		}
		this->threads.clear();

		predict = this->layers[this->lastIndex]->getOutputArray1D();
		for (int x = 0; x < this->data->getImgsNum() % this->batchSize; x++) {
			max = predict[x][0];
			maxIndex = 0;
			for (int l = 0; l < labelsNum; l++) {
				if (predict[x][l] > max) {
					max = predict[x][l];
					maxIndex = l;
				}
			}
			ret += "Plik: " + imgsPath[batchesNum * this->batchSize + x] + "\nPredykcja: " + labels[maxIndex] + '\n';
			for (int l = 0; l < labelsNum; l++) {
				ret += labels[l] + ": " + std::to_string(predict[x][l] * 100.0f) + "%\n";
			}
		}
		return ret;
	}

	void createModelFromVector(std::vector<std::vector<int>> layers) {
		for (size_t x = 0; x < layers.size(); x++) {
			if (layers[x][0] == 0) {
				this->add(new Input(layers[x][1], layers[x][2], layers[x][3]));
			}
			else if (layers[x][0] == 1) {
				this->add(new Conv(layers[x][1], layers[x][2], layers[x][3]));
			}
			else if (layers[x][0] == 2) {
				this->add(new Flatten());
			}
			else if (layers[x][0] == 3) {
				this->add(new Dense(layers[x][1]));
			}
		}
	}
	bool loadWeight(std::string weightsPath) {
		nlohmann::json j;
		std::ifstream file(weightsPath);
		if (file.is_open()) {
			file >> j;
			file.close();
			if (j.is_array()) {
				int weightCount = 0;
				for (int x = 0; x <= this->lastIndex; x++) {
					weightCount += this->layers[x]->getWeightCount();
				}
				if (j.size() == weightCount) {
					int targetIndex = 0;
					for (int x = 0; x <= this->lastIndex; x++) {
						if (this->layers[x]->getWeightCount() == 2) {
							if (this->layers[x]->setWeight(j[targetIndex], j[targetIndex + 1])) {
								targetIndex += 2;
							}
							else {
								std::cout << "Nieprawidlowe wagi" << std::endl;
								return false;
							}
						}
					}
					return true;
				}
				else {
					std::cout << "Nieprawidlowe wagi" << std::endl;
					return false;
				}
			}
			else {
				std::cout << "Nieprawidlowe wagi" << std::endl;
				return false;
			}
		}
		else {
			std::cout << "Nie mozna otworzyc pliku z wagami" << std::endl;
			return false;
		}
	}
};