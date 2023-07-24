#include "json.hpp"
#include <random>
#include <cmath>

export import Layer;

export module Conv;


export class Conv : public Layer {
private:
	int batchSize = -1;
	int weightY = -1;
	int weightX = -1;
	int weightC = -1;
	int weightK = -1;
	int outputY = -1;
	int outputX = -1;
	int outputC = -1;
	int biasX = -1;

	float learningRate = -1.0f;
	Layer* prev = nullptr;

	std::atomic<float****> dWeight = nullptr;
	std::atomic<float*> dBias = nullptr;
	float**** outputArray = nullptr;
	float**** weight = nullptr;
	float**** dB = nullptr;
	float* bias = nullptr;
public:
	Conv(int K, int Y, int X) {
		this->weightY = Y;
		this->weightX = X;
		this->weightK = K;
		this->outputC = K;
		this->biasX = K;
	}
	~Conv() {
		for (int bs = 0; bs < this->batchSize; bs++) {
			for (int y = 0; y < this->outputY; y++) {
				for (int x = 0; x < this->outputX; x++) {
					delete[] this->dB[bs][y][x];
					delete[] this->outputArray[bs][y][x];
				}
				delete[] this->dB[bs][y];
				delete[] this->outputArray[bs][y];
			}
			delete[] this->dB[bs];
			delete[] this->outputArray[bs];
		}
		delete[] this->dB;
		delete[] this->outputArray;

		delete[] this->bias;
		delete[] this->dBias;

		for (int y = 0; y < this->weightY; y++) {
			for (int x = 0; x < this->weightX; x++) {
				for (int c = 0; c < this->weightC; c++) {
					delete[] this->weight[y][x][c];
					delete[] this->dWeight[y][x][c];
				}
				delete[] this->weight[y][x];
				delete[] this->dWeight[y][x];
			}
			delete[] this->weight[y];
			delete[] this->dWeight[y];
		}
		delete[] this->weight;
		delete[] this->dWeight;
	}
	bool compile(Layer* prev, float learningRate, int batchSize) {
		if (prev->getOutputDimsCount() == 3 && this->weightY <= prev->getOutputY() && this->weightX <= prev->getOutputX()) {
			this->prev = prev;
			this->outputY = prev->getOutputY() - this->weightY + 1;
			this->outputX = prev->getOutputX() - this->weightX + 1;
			this->weightC = prev->getOutputC();
			this->learningRate = learningRate;
			this->batchSize = batchSize;

			float limit = sqrt(6.0f / ((float)this->weightY * (float)this->weightX * (float)this->weightC + (float)this->weightY * (float)this->weightX) * (float)this->weightK);
			std::random_device rd{};
			std::mt19937 gen{ rd() };
			std::uniform_real_distribution<> d(-limit, limit);

			this->dWeight = new float*** [this->weightY];
			for (int y = 0; y < this->weightY; y++) {
				this->dWeight[y] = new float** [this->weightX];
				for (int x = 0; x < this->weightX; x++) {
					this->dWeight[y][x] = new float* [this->weightC];
					for (int c = 0; c < this->weightC; c++) {
						this->dWeight[y][x][c] = new float[this->weightK];
						for (int k = 0; k < this->weightK; k++) {
							this->dWeight[y][x][c][k] = 0.0f;
						}
					}
				}
			}

			this->dBias = new float[this->weightK];
			for (int k = 0; k < this->weightK; k++) {
				this->dBias[k] = 0.0f;
			}

			this->weight = new float*** [this->weightY];
			for (int y = 0; y < this->weightY; y++) {
				this->weight[y] = new float** [this->weightX];
				for (int x = 0; x < this->weightX; x++) {
					this->weight[y][x] = new float* [this->weightC];
					for (int c = 0; c < this->weightC; c++) {
						this->weight[y][x][c] = new float[this->weightK];
						for (int k = 0; k < this->weightK; k++) {
							this->weight[y][x][c][k] = d(gen);
						}
					}
				}
			}

			this->bias = new float[this->weightK];
			for (int k = 0; k < this->weightK; k++) {
				this->bias[k] = 0.0f;
			}

			this->outputArray = new float*** [batchSize];
			for (int bs = 0; bs < batchSize; bs++) {
				this->outputArray[bs] = new float** [this->outputY];
				for (int y = 0; y < this->outputY; y++) {
					this->outputArray[bs][y] = new float* [this->outputX];
					for (int x = 0; x < this->outputX; x++) {
						this->outputArray[bs][y][x] = new float[this->outputC];
					}
				}
			}

			this->dB = new float*** [batchSize];
			for (int bs = 0; bs < batchSize; bs++) {
				this->dB[bs] = new float** [this->outputY];
				for (int y = 0; y < this->outputY; y++) {
					this->dB[bs][y] = new float* [this->outputX];
					for (int x = 0; x < this->outputX; x++) {
						this->dB[bs][y][x] = new float[this->outputC];
					}
				}
			}

			return true;
		}
		else {
			return false;
		}
	}
	int getOutputDimsCount() {
		return 3;
	}
	int getOutputY() {
		return this->outputY;
	}
	int getOutputX() {
		return this->outputX;
	}
	int getOutputC() {
		return this->outputC;
	}
	float**** getOutputArray3D() {
		return this->outputArray;
	}

	void forward(int stepNum) {
		float**** prevOutputArray = prev->getOutputArray3D();
		float sum;
		float cache;
		for (int k = 0; k < this->weightK; k++) {
			for (int y = 0; y < this->outputY; y++) {
				for (int x = 0; x < this->outputX; x++) {
					sum = 0.0f;
					for (int c = 0; c < this->weightC; c++) {
						cache = 0.0f;
						for (int y1 = 0; y1 < this->weightY; y1++) {
							for (int x1 = 0; x1 < this->weightX; x1++) {
								cache += prevOutputArray[stepNum][y + y1][x + x1][c] * this->weight[y1][x1][c][k];
							}
						}
						sum += cache;
					}
					this->outputArray[stepNum][y][x][k] = sigmoid(sum + this->bias[k]);
				}
			}
		}
	}

	float**** getdB3D() {
		return this->dB;
	}
	void calcPrevdB(int stepNum) {
		float**** prevOutputArray = this->prev->getOutputArray3D();
		float**** prevdB = this->prev->getdB3D();
		int prevOutputY = this->prev->getOutputY();
		int prevOutputX = this->prev->getOutputX();

		float cache;
		int rY;
		int rX;
		for (int c = 0; c < this->weightC; c++) {
			for (int y = 0; y < prevOutputY; y++) {
				for (int x = 0; x < prevOutputX; x++) {
					cache = 0.0f;
					for (int k = 0; k < this->weightK; k++) {
						for (int y1 = 0; y1 < this->weightY; y1++) {
							rY = y - this->weightY + 1 + y1;
							if (rY > -1 && rY < this->outputY) {
								for (int x1 = 0; x1 < this->weightX; x1++) {
									rX = x - this->weightX + 1 + x1;
									if (rX > -1 && rX < this->outputX) {
										cache += this->dB[stepNum][rY][rX][k] * this->weight[this->weightY - y1 - 1][this->weightX - x1 - 1][c][k];
									}
								}
							}
						}
					}
					prevdB[stepNum][y][x][c] = cache * (prevOutputArray[stepNum][y][x][c] * (1.0f - prevOutputArray[stepNum][y][x][c]));
				}
			}
		}
	}
	void calcdW(int stepNum) {
		float**** prevOutputArray = this->prev->getOutputArray3D();
		float cache;

		for (int c = 0; c < this->weightC; c++) {
			for (int k = 0; k < this->weightK; k++) {
				for (int y = 0; y < this->weightY; y++) {
					for (int x = 0; x < this->weightX; x++) {
						cache = 0.0f;
						for (int y1 = 0; y1 < this->outputY; y1++) {
							for (int x1 = 0; x1 < this->outputX; x1++) {
								cache += prevOutputArray[stepNum][y + y1][x + x1][c] * this->dB[stepNum][y1][x1][k];
							}
						}
						this->dWeight[y][x][c][k] += cache;
					}
				}
			}
		}

		for (int c = 0; c < this->outputC; c++) {
			cache = 0.0f;
			for (int y = 0; y < this->outputY; y++) {
				for (int x = 0; x < this->outputX; x++) {
					cache += this->dB[stepNum][y][x][c];
				}
			}
			this->dBias[c] += cache;
		}
	}
	void updateWeights(int batchSize) {
		for (int y = 0; y < this->weightY; y++) {
			for (int x = 0; x < this->weightX; x++) {
				for (int c = 0; c < this->weightC; c++) {
					for (int k = 0; k < this->weightK; k++) {
						this->weight[y][x][c][k] -= this->dWeight[y][x][c][k] / (float)batchSize * this->learningRate;
					}
				}
			}
		}

		for (int k = 0; k < this->weightK; k++) {
			this->bias[k] -= this->dBias[k] / (float)batchSize * this->learningRate;
		}
	}
	void setdWToZero() {
		for (int y = 0; y < this->weightY; y++) {
			for (int x = 0; x < this->weightX; x++) {
				for (int c = 0; c < this->weightC; c++) {
					for (int k = 0; k < this->weightK; k++) {
						this->dWeight[y][x][c][k] = 0.0f;;
					}
				}
			}
		}

		for (int k = 0; k < this->weightK; k++) {
			this->dBias[k] = 0.0f;
		}
	}

	nlohmann::json getWeight() {
		nlohmann::json weight = nlohmann::json::array();
		for (int y = 0; y < this->weightY; y++) {
			nlohmann::json cacheY = nlohmann::json::array();
			for (int x = 0; x < this->weightX; x++) {
				nlohmann::json cacheX = nlohmann::json::array();
				for (int c = 0; c < this->weightC; c++) {
					nlohmann::json cacheC = nlohmann::json::array();
					for (int k = 0; k < this->weightK; k++) {
						cacheC.push_back(this->weight[y][x][c][k]);
					}
					cacheX.push_back(cacheC);
				}
				cacheY.push_back(cacheX);
			}
			weight.push_back(cacheY);
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
	bool setWeight(nlohmann::json weight, nlohmann::json bias) {
		if (weight.is_array() and bias.is_array()) {
			if (weight.size() == this->weightY) {
				if (weight[0].is_array()) {
					if (weight[0].size() == this->weightX) {
						if (weight[0][0].is_array()) {
							if (weight[0][0].size() == this->weightC) {
								if (weight[0][0][0].is_array()) {
									if (weight[0][0][0].size() == this->weightK) {
										for (int y = 0; y < this->weightY; y++) {
											for (int x = 0; x < this->weightX; x++) {
												for (int c = 0; c < this->weightC; c++) {
													for (int k = 0; k < this->weightK; k++) {
														if (weight[y][x][c][k].is_number()) {
															this->weight[y][x][c][k] = weight[y][x][c][k];
														}
														else {
															return false;
														}
													}
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
};