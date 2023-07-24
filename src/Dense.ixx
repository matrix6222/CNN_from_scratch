#include <iostream>
#include "json.hpp"
#include <vector>
#include <random>
#include <cmath>

export import Layer;

export module Dense;


export class Dense : public Layer {
private:
	int batchSize = -1;
	int weightY = -1;
	int weightX = -1;
	int outputX = -1;
	int biasX = -1;

	float learningRate = -1.0f;
	Layer* prev = nullptr;

	std::atomic<float**> dWeight = nullptr;
	std::atomic<float*> dBias = nullptr;
	float** outputArray = nullptr;
	float** weight = nullptr;
	float* bias = nullptr;
	float** dB = nullptr;
public:
	Dense(int X) {
		this->outputX = X;
	}
	~Dense() {
		for (int bs = 0; bs < batchSize; bs++) {
			delete[] this->dB[bs];
			delete[] this->outputArray[bs];
		}
		delete[] this->dB;
		delete[] this->outputArray;

		delete[] this->bias;
		delete[] this->dBias;

		for (int y = 0; y < this->weightY; y++) {
			delete[] this->dWeight[y];
			delete[] this->weight[y];
		}
		delete[] this->dWeight;
		delete[] this->weight;
	}
	bool compile(Layer* prev, float learningRate, int batchSize) {
		if (prev->getOutputDimsCount() == 1) {
			this->prev = prev;
			this->weightX = this->outputX;
			this->weightY = prev->getOutputX();
			this->biasX = this->outputX;
			this->outputX = this->outputX;
			this->learningRate = learningRate;
			this->batchSize = batchSize;

			
			float limit = sqrt(6.0f / ((float)this->weightY + (float)this->weightX));
			std::random_device rd{};
			std::mt19937 gen{ rd() };
			std::uniform_real_distribution<> d(-limit, limit);

			this->dWeight = new float* [this->weightY];
			for (int y = 0; y < this->weightY; y++) {
				this->dWeight[y] = new float[this->weightX];
				for (int x = 0; x < this->weightX; x++) {
					this->dWeight[y][x] = 0.0f;
				}
			}

			this->dBias = new float[this->biasX];
			for (int x = 0; x < this->biasX; x++) {
				this->dBias[x] = 0.0f;
			}

			this->weight = new float* [this->weightY];
			for (int y = 0; y < this->weightY; y++) {
				this->weight[y] = new float[this->weightX];
				for (int x = 0; x < this->weightX; x++) {
					this->weight[y][x] = d(gen);
				}
			}

			this->bias = new float[this->biasX];
			for (int x = 0; x < this->biasX; x++) {
				this->bias[x] = 0.0f;
			}

			this->outputArray = new float*[batchSize];
			for (int bs = 0; bs < batchSize; bs++) {
				this->outputArray[bs] = new float[this->outputX];
			}

			this->dB = new float*[batchSize];
			for (int bs = 0; bs < batchSize; bs++) {
				this->dB [bs] = new float[this->outputX];
			}
			
			return true;
		}
		else {
			return false;
		}
	}
	float** getOutputArray1D() {
		return this->outputArray;
	}
	int getOutputDimsCount() {
		return 1;
	}
	int getOutputX() {
		return this->outputX;
	}

	void forward(int stepNum) {
		float** prevOutputArray = prev->getOutputArray1D();

		float cache;
		for (int x = 0; x < this->outputX; x++) {
			cache = 0.0;
			for (int y = 0; y < this->weightY; y++) {
				cache += prevOutputArray[stepNum][y] * this->weight[y][x];
			}
			this->outputArray[stepNum][x] = this->sigmoid(cache + this->bias[x]);
		}
	}

	float** getdB1D() {
		return this->dB;
	}
	void calcPrevdB(int stepNum) {
		float** prevdB = this->prev->getdB1D();
		float** prevOutputArray = this->prev->getOutputArray1D();

		float cache;
		for (int x = 0; x < this->weightY; x++) {
			cache = 0.0;
			for (int y = 0; y < this->weightX; y++) {
				cache += this->dB[stepNum][y] * this->weight[x][y];
			}
			prevdB[stepNum][x] = cache * prevOutputArray[stepNum][x] * (1.0f - prevOutputArray[stepNum][x]);
		}
	}
	void calcdW(int stepNum) {
		float** prevOutputArray = this->prev->getOutputArray1D();

		for (int y = 0; y < this->weightY; y++) {
			for (int x = 0; x < this->weightX; x++) {
				this->dWeight[y][x] += this->dB[stepNum][x] * prevOutputArray[stepNum][y];
			}
		}

		for (int x = 0; x < this->biasX; x++) {
			this->dBias[x] += this->dB[stepNum][x];
		}
	}
	void updateWeights(int batchSzie) {
		for (int y = 0; y < this->weightY; y++) {
			for (int x = 0; x < this->weightX; x++) {
				this->weight[y][x] -= this->dWeight[y][x] / (float)batchSzie * learningRate;
			}
		}

		for (int x = 0; x < this->biasX; x++) {
			this->bias[x] -= this->dBias[x] / (float)batchSzie * learningRate;
		}
	}
	void setdWToZero() {
		for (int y = 0; y < this->weightY; y++) {
			for (int x = 0; x < this->weightX; x++) {
				this->dWeight[y][x] = 0.0f;
			}
		}

		for (int x = 0; x < this->biasX; x++) {
			this->dBias[x] = 0.0f;
		}
	}

	
	bool setWeight(nlohmann::json weight, nlohmann::json bias) {
		if (weight.is_array() and bias.is_array()) {
			if (weight.size() == this->weightY) {
				if (weight[0].is_array()) {
					if (weight[0].size() == this->weightX) {
						for (int y = 0; y < this->weightY; y++) {
							for (int x = 0; x < this->weightX; x++) {
								if (weight[y][x].is_number()) {
									this->weight[y][x] = weight[y][x];
								}
								else {
									return false;
								}
							}
						}
					}
					else {
						return false;
					}
				}
				else {
					return false;
				}
			}
			else {
				return false;
			}

			if (bias.size() == this->biasX) {
				for (int x = 0; x < this->biasX; x++) {
					if (bias[x].is_number()) {
						this->bias[x] = bias[x];
					}
					else {
						return false;
					}
				}
			}
			else {
				return false;
			}

			return true;
		}
		else {
			return false;
		}
	}
	nlohmann::json getWeight() {
		nlohmann::json weight = nlohmann::json::array();
		for (int y = 0; y < this->weightY; y++) {
			nlohmann::json cache = nlohmann::json::array();
			for (int x = 0; x < this->weightX; x++) {
				cache.push_back(this->weight[y][x]);
			}
			weight.push_back(cache);
		}

		nlohmann::json bias = nlohmann::json::array();
		for (int x = 0; x < this->biasX; x++) {
			bias.push_back(this->bias[x]);
		}

		nlohmann::json ret = nlohmann::json::array();
		ret.push_back(weight);
		ret.push_back(bias);

		return ret;
	}
	int getWeightCount() {
		return 2;
	}
};