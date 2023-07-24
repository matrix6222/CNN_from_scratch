#include <iostream>
#include "json.hpp"
#include <fstream>
#include <vector>

import Params;

export module Config;


export class Config {
private:
	Params params = Params();

	int imgY = -1;
	int imgX = -1;
	int imgC = -1;

	std::string trainXPath;
	std::string trainYPath;
	std::string testXPath;
	std::string testYPath;
	std::string labelsPath;

	std::string weigthsPath;

	int batchSize = -1;

	int trainNum = -1;
	int testNum = -1;
	int labelsNum = -1;

	float learningRate = -1.0f;

	int epochs = -1;

	std::vector<std::vector<int>> layers;

	bool readImgY(nlohmann::json j) {
		try {
			this->imgY = j["imgY"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}
	bool readImgX(nlohmann::json j) {
		try {
			this->imgX = j["imgX"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}
	bool readImgC(nlohmann::json j) {
		try {
			this->imgC = j["imgC"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}

	bool readTrainXPath(nlohmann::json j) {
		try {
			this->trainXPath = j["trainX"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}
	bool readTrainYPath(nlohmann::json j) {
		try {
			this->trainYPath = j["trainY"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}
	bool readTestXPath(nlohmann::json j) {
		try {
			this->testXPath = j["testX"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}
	bool readTestYPath(nlohmann::json j) {
		try {
			this->testYPath = j["testY"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}
	bool readLabelsPath(nlohmann::json j) {
		try {
			this->labelsPath = j["labels"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}

	bool readWeightsPath(nlohmann::json j) {
		try {
			this->weigthsPath = j["weights"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}

	bool readBatchSize(nlohmann::json j) {
		try {
			this->batchSize = j["batchSize"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}

	bool readTrainNum(nlohmann::json j) {
		try {
			this->trainNum = j["trainNum"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}
	bool readTestNum(nlohmann::json j) {
		try {
			this->testNum = j["testNum"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}
	bool readLabelsNum(nlohmann::json j) {
		try {
			this->labelsNum = j["labelsNum"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}

	bool readLearningRate(nlohmann::json j) {
		try {
			this->learningRate = j["learningRate"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}

	bool readEpochs(nlohmann::json j) {
		try {
			this->epochs = j["epochs"];
		}
		catch (nlohmann::json::type_error) {
			return false;
		}
		return true;
	}

	bool readLayers(nlohmann::json j) {
		if (j["Layers"].is_array()) {
			for (size_t x = 0; x < j["Layers"].size(); x++) {

				if (j["Layers"][x].size() > 0) {
					if (j["Layers"][x][0] == "Input") {
						if (j["Layers"][x].size() == 4) {
							std::vector<int> cache;
							try {
								cache.push_back(0);
								cache.push_back(j["Layers"][x][1]);
								cache.push_back(j["Layers"][x][2]);
								cache.push_back(j["Layers"][x][3]);
							}
							catch (nlohmann::json::type_error) {
								return false;
							}
							this->layers.push_back(cache);
						}
						else {
							return false;
						}
					}
					else if (j["Layers"][x][0] == "Conv") {
						if (j["Layers"][x].size() == 4) {
							std::vector<int> cache;
							try {
								cache.push_back(1);
								cache.push_back(j["Layers"][x][1]);
								cache.push_back(j["Layers"][x][2]);
								cache.push_back(j["Layers"][x][3]);
							}
							catch (nlohmann::json::type_error) {
								return false;
							}
							this->layers.push_back(cache);
						}
						else {
							return false;
						}
					}
					else if (j["Layers"][x][0] == "Flatten") {
						if (j["Layers"][x].size() == 1) {
							std::vector<int> cache;
							cache.push_back(2);
							this->layers.push_back(cache);
						}
						else {
							return false;
						}
					}
					else if (j["Layers"][x][0] == "Dense") {
						if (j["Layers"][x].size() == 2) {
							std::vector<int> cache;
							try {
								cache.push_back(3);
								cache.push_back(j["Layers"][x][1]);
							}
							catch (nlohmann::json::type_error) {
								return false;
							}
							this->layers.push_back(cache);
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
			if (this->layers.size() > 0) {
				if (this->layers[0].size() == 4) {
					if (this->layers[0][0] == 0 && this->layers[0][1] == this->imgY && this->layers[0][2] == this->imgX && this->layers[0][3] == this->imgC) {
						return true;
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
public:
	Config() {
	}
	bool loadConfig(const int& argc, const char* argv[]) {
		if (this->params.loadParams(argc, argv)) {
			int mode = this->params.getMode();
			if (mode == 0 || mode == 1) {
				std::ifstream file(params.getConfigPath());
				if (file.is_open()) {
					nlohmann::json j;
					try {
						file >> j;
					}
					catch (nlohmann::json::parse_error) {
						std::cout << "Plik konfiguracyjny niezgodny z formatem JSON" << std::endl;
						return false;
					}
					file.close();

					if (this->readImgY(j) && this->readImgX(j) && this->readImgC(j) && this->readWeightsPath(j) && this->readBatchSize(j) && this->readLabelsNum(j) && this->readLabelsPath(j)) {
						if (this->readLayers(j)) {
							return true;
						}
						else {
							std::cout << "Nieprawidlowe warstwy" << std::endl;
							return false;
						}
					}
					else {
						std::cout << "Nieprawidlowe parametry pliku konfiguracyjnego" << std::endl;
						return false;
					}
				}
				else {
					std::cout << "Nie mozna otworzyc pliku konfiguracyjnego" << std::endl;
					return false;
				}
			}
			else if (mode == 2) {
				return true;
			}
			else if (mode == 3 || mode == 4) {
				std::ifstream file(params.getConfigPath());
				if (file.is_open()) {
					nlohmann::json j;
					try {
						file >> j;
					}
					catch (nlohmann::json::parse_error) {
						std::cout << "Plik konfiguracyjny niezgodny z formatem JSON" << std::endl;
						return false;
					}
					file.close();

					if (this->readImgY(j) && this->readImgX(j) && this->readImgC(j) && this->readTrainXPath(j) && this->readTrainYPath(j) && this->readTestXPath(j) && this->readTestYPath(j) && this->readLabelsPath(j) && this->readWeightsPath(j) && this->readBatchSize(j) && this->readTrainNum(j) && this->readTestNum(j) && this->readLabelsNum(j) && this->readLearningRate(j) && this->readEpochs(j)) {
						if (this->readLayers(j)) {
							if (this->trainNum % this->batchSize == 0 && this->testNum % this->batchSize == 0) {
								return true;
							}
							else {
								std::cout << "TrainNum i testNum nie sa podzielne przez batchSize" << std::endl;
								return false;
							}
						}
						else {
							std::cout << "Nieprawidlowe warstwy" << std::endl;
							return false;
						}
					}
					else {
						std::cout << "Nieprawidlowe parametry pliku konfiguracyjnego" << std::endl;
						return false;
					}
				}
				else {
					std::cout << "Nie mozna otworzyc pliku konfiguracyjnego" << std::endl;
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

	Params getParams() {
		return this->params;
	}

	int getImgY() {
		return this->imgY;
	}
	int getImgX() {
		return this->imgX;
	}
	int getImgC() {
		return this->imgC;
	}

	std::string getTrainXPath() {
		return this->trainXPath;
	}
	std::string getTrainYPath() {
		return this->trainYPath;
	}
	std::string getTestXPath() {
		return this->testXPath;
	}
	std::string getTestYPath() {
		return this->testYPath;
	}
	std::string getLabelsPath() {
		return this->labelsPath;
	}

	std::string getWeigthsPath() {
		return this->weigthsPath;
	}

	int getBatchSize() {
		return this->batchSize;
	}

	int getTrainNum() {
		return this->trainNum;
	}
	int getTestNum() {
		return this->testNum;
	}
	int getLabelsNum() {
		return this->labelsNum;
	}

	float getLearningRate() {
		return learningRate;
	}

	int getEpochs() {
		return this->epochs;
	}

	std::vector<std::vector<int>> getLayers() {
		return this->layers;
	}
};