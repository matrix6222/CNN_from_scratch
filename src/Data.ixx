#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>

export module Data;


export class Data {
private:
	int labelsNum = -1;
	int trainNum = -1;
	int testNum = -1;
	int imgsNum = 0;
	int imgY = -1;
	int imgX = -1;
	int imgC = -1;
	
	float**** trainX = nullptr;
	float**** testX = nullptr;
	float**** imgs = nullptr;
	float** trainY = nullptr;
	float** testY = nullptr;

	std::vector<std::string> imgsPath;
	std::vector<std::string> labels;
public:
	Data(int imgY, int imgX, int imgC, int trainNum, int testNum, int labelsNum) {
		this->labelsNum = labelsNum;
		this->trainNum = trainNum;
		this->testNum = testNum;
		this->imgY = imgY;
		this->imgX = imgX;
		this->imgC = imgC;
	}
	Data(int imgY, int imgX, int imgC, int labelsNum) {
		this->labelsNum = labelsNum;
		this->imgY = imgY;
		this->imgX = imgX;
		this->imgC = imgC;
	}
	Data() {
	}
	~Data() {
		for (int i = 0; i < this->trainNum; i++) {
			for (int y = 0; y < this->imgY; y++) {
				for (int x = 0; x < this->imgX; x++) {
					delete[] this->trainX[i][y][x];
				}
				delete[] this->trainX[i][y];
			}
			delete[] this->trainX[i];
		}
		delete[] this->trainX;

		for (int i = 0; i < this->testNum; i++) {
			for (int y = 0; y < this->imgY; y++) {
				for (int x = 0; x < this->imgX; x++) {
					delete[] this->testX[i][y][x];
				}
				delete[] this->testX[i][y];
			}
			delete[] this->testX[i];
		}
		delete[] this->testX;

		for (int i = 0; i < this->imgsNum; i++) {
			for (int y = 0; y < this->imgY; y++) {
				for (int x = 0; x < this->imgX; x++) {
					delete[] this->imgs[i][y][x];
				}
				delete[] this->imgs[i][y];
			}
			delete[] this->imgs[i];
		}
		delete[] this->imgs;

		for (int y = 0; y < this->trainNum; y++) {
			delete[] this->trainY[y];
		}
		delete[] this->trainY;

		for (int y = 0; y < this->testNum; y++) {
			delete[] this->testY[y];
		}
		delete[] this->testY;
	}

	bool loadTrainX(std::filesystem::path path) {
		if (std::filesystem::is_directory(path)) {
			this->trainX = new float***[this->trainNum];
			for (int i = 0; i < this->trainNum; i++) {
				this->trainX[i] = new float** [this->imgY];
				for (int y = 0; y < this->imgY; y++) {
					this->trainX[i][y] = new float* [this->imgX];
					for (int x = 0; x < this->imgX; x++) {
						this->trainX[i][y][x] = new float[this->imgC];
					}
				}
			}
			int count = 0;
			int Y;
			int X;
			int C;
			int XC = this->imgX * this->imgC;
			for (int i = 0; i < this->trainNum; i++) {
				unsigned char* img = stbi_load((path / (std::to_string(i) + ".png")).string().c_str(), &X, &Y, &C, this->imgC);
				if (img != NULL && Y == this->imgY && X == this->imgX && C == this->imgC) {
					for (int y = 0; y < Y; y++) {
						for (int x = 0; x < X; x++) {
							for (int c = 0; c < C; c++) {
								this->trainX[i][y][x][c] = (float)img[y * XC + x * C + c] / 255.0f;
							}
						}
					}
					stbi_image_free(img);
				}
				else {
					stbi_image_free(img);
					std::cout << "Nieprawidlowy obraz trainX " << i << ".png" << std::endl;
					return false;
				}
			}
		}
		else {
			std::cout << "Nieprawidlowa sciezka do folderu trainX" << std::endl;
			return false;
		}
	}
	bool loadTrainY(std::filesystem::path path) {
		if (std::filesystem::is_regular_file(path)) {
			std::ifstream file(path);
			if (file.is_open()) {
				int count = 0;
				int ID = 0;
				this->trainY = new float* [this->trainNum];
				for (int y = 0; y < this->trainNum; y++) {
					this->trainY[y] = new float[this->labelsNum];
					for (int x = 0; x < this->labelsNum; x++) {
						this->trainY[y][x] = 0.0f;
					}
				}
				while (file >> ID && count < this->trainNum) {
					if (ID >= 0 && ID < this->labelsNum) {
						trainY[count][ID] = 1.0f;
						count++;
					}
					else {
						file.close();
						std::cout << "Nieprawidlowy index etykiety" << std::endl;
						return false;
					}
				}
				file.close();
				if (this->trainNum == count) {
					return true;
				}
				else {
					std::cout << "Nieprawidlowa ilosc danych trainY" << std::endl;
					return false;
				}
			}
			else {
				std::cout << "Nie mozna otworzyc pliku z danymi trainY" << std::endl;
				return false;
			}
		}
		else {
			std::cout << "Nieprawidlowa sciezka do pliku trainY" << std::endl;
			return false;
		}
	}
	bool loadTestX(std::filesystem::path path) {
		if (std::filesystem::is_directory(path)) {
			this->testX = new float*** [this->testNum];
			for (int i = 0; i < this->testNum; i++) {
				this->testX[i] = new float** [this->imgY];
				for (int y = 0; y < this->imgY; y++) {
					this->testX[i][y] = new float* [this->imgX];
					for (int x = 0; x < this->imgX; x++) {
						this->testX[i][y][x] = new float[this->imgC];
					}
				}
			}
			int count = 0;
			int Y;
			int X;
			int C;
			int XC = this->imgX * this->imgC;
			for (int i = 0; i < this->testNum; i++) {
				unsigned char* img = stbi_load((path / (std::to_string(i) + ".png")).string().c_str(), &X, &Y, &C, this->imgC);
				if (img != NULL && Y == this->imgY && X == this->imgX && C == this->imgC) {
					for (int y = 0; y < Y; y++) {
						for (int x = 0; x < X; x++) {
							for (int c = 0; c < C; c++) {
								this->testX[i][y][x][c] = (float)img[y * XC + x * C + c] / 255.0f;
							}
						}
					}
					stbi_image_free(img);
				}
				else {
					stbi_image_free(img);
					std::cout << "Nieprawidlowy obraz testX " << i << ".png" << std::endl;
					return false;
				}
			}
		}
		else {
			std::cout << "Nieprawidlowa sciezka do folderu testX" << std::endl;
			return false;
		}
	}
	bool loadTestY(std::filesystem::path path) {
		if (std::filesystem::is_regular_file(path)) {
			std::ifstream file(path);
			if (file.is_open()) {
				int count = 0;
				int ID = 0;
				this->testY = new float* [this->testNum];
				for (int y = 0; y < this->testNum; y++) {
					this->testY[y] = new float[this->labelsNum];
					for (int x = 0; x < this->labelsNum; x++) {
						this->testY[y][x] = 0.0f;
					}
				}
				while (file >> ID && count < this->testNum) {
					if (ID >= 0 && ID < this->labelsNum) {
						testY[count][ID] = 1.0f;
						count++;
					}
					else {
						file.close();
						std::cout << "Nieprawidlowy index etykiety" << std::endl;
						return false;
					}
				}
				file.close();
				if (this->testNum == count) {
					return true;
				}
				else {
					std::cout << "Nieprawidlowa ilosc danych testY" << std::endl;
					return false;
				}
			}
			else {
				std::cout << "Nie mozna otworzyc pliku z danymi testY" << std::endl;
				return false;
			}
		}
		else {
			std::cout << "Nieprawidlowa sciezka do pliku testY" << std::endl;
			return false;
		}
	}
	bool loadLabels(std::filesystem::path path) {
		if (std::filesystem::is_regular_file(path)) {
			std::ifstream file(path);
			if (file.is_open()) {
				int count = 0;
				std::string name;
				while (file >> name) {
					this->labels.push_back(name);
					count++;
				}
				file.close();
				if (this->labelsNum == count) {
					return true;
				}
				else {
					std::cout << "Nieprawidlowa ilosc etykiet" << std::endl;
					return false;
				}
			}
			else {
				std::cout << "Nie mozna otworzyc pliku z etykietami" << std::endl;
				return false;
			}
		}
		else {
			std::cout << "Nieprawidlowa sciezka do pliku z etykietami" << std::endl;
			return false;
		}
	}
	bool loadImgs(std::filesystem::path path) {
		if (std::filesystem::is_directory(path)) {
			std::vector<std::vector<std::vector<std::vector<float>>>> imgsVec;
			int Y;
			int X;
			int C;
			int XC = this->imgX * this->imgC;
			for (const std::filesystem::directory_entry& dir_entry : std::filesystem::directory_iterator(path)) {
				if (std::filesystem::is_regular_file(dir_entry)) {
					unsigned char* img = stbi_load(dir_entry.path().string().c_str(), &X, &Y, &C, this->imgC);
					if (img != NULL && Y == this->imgY && X == this->imgX && C == this->imgC) {
						this->imgsPath.push_back(dir_entry.path().string());
						std::vector<std::vector<std::vector<float>>> cacheY;
						for (int y = 0; y < Y; y++) {
							std::vector<std::vector<float>> cacheX;
							for (int x = 0; x < X; x++) {
								std::vector<float> cacheC;
								for (int c = 0; c < C; c++) {
									cacheC.push_back((float)img[y * XC + x * C + c] / 255.0f);
								}
								cacheX.push_back(cacheC);
							}
							cacheY.push_back(cacheX);
						}
						imgsVec.push_back(cacheY);
					}
					stbi_image_free(img);
				}
			}
			this->imgsNum = imgsVec.size();
			this->imgs = new float*** [this->imgsNum];
			for (int i = 0; i < this->imgsNum; i++) {
				this->imgs[i] = new float** [this->imgY];
				for (int y = 0; y < this->imgY; y++) {
					this->imgs[i][y] = new float* [this->imgX];
					for (int x = 0; x < this->imgX; x++) {
						this->imgs[i][y][x] = new float[this->imgC];
						for (int c = 0; c < this->imgC; c++) {
							this->imgs[i][y][x][c] = imgsVec[i][y][x][c];
						}
					}
				}
			}
		}
		else {
			std::cout << "Nieprawidlowa sciezka do folderu z obrazami" << std::endl;
			return false;
		}
	}

	bool loadImageFromArray(std::array<unsigned char, 65535> arr, int len) {
		unsigned char* buf = new unsigned char[65535];
		for (int x = 0; x < len; x++) {
			buf[x] = arr[x];
		}
		int Y;
		int X;
		int C;
		int XC = this->imgX * this->imgC;
		unsigned char* img = stbi_load_from_memory(buf, len, &X, &Y, &C, this->imgC);
		delete[] buf;
		if (img != NULL && Y == this->imgY && X == this->imgX && C == this->imgC) {
			if (this->imgs == nullptr) {
				this->imgsPath.push_back("Twoj plik");
				this->imgsNum = 1;
				this->imgs = new float***[this->imgsNum];
				for (int i = 0; i < this->imgsNum; i++) {
					this->imgs[i] = new float** [this->imgY];
					for (int y = 0; y < this->imgY; y++) {
						this->imgs[i][y] = new float* [this->imgX];
						for (int x = 0; x < this->imgX; x++) {
							this->imgs[i][y][x] = new float[this->imgC];
							for (int c = 0; c < this->imgC; c++) {
								this->imgs[i][y][x][c] = (float)img[y * XC + x * C + c] / 255.0f;
							}
						}
					}
				}
				return true;
			}
			else {
				for (int i = 0; i < this->imgsNum; i++) {
					for (int y = 0; y < this->imgY; y++) {
						for (int x = 0; x < this->imgX; x++) {
							for (int c = 0; c < this->imgC; c++) {
								this->imgs[i][y][x][c] = (float)img[y * XC + x * C + c] / 255.0f;
							}
						}
					}
				}
				return true;
			}
		}
		else {
			std::cout << "Klient wyslal niepoprawny obrazek" << std::endl;
			return false;
		}
	}

	std::vector<std::string> getImgsPath() {
		return this->imgsPath;
	}
	std::vector<std::string> getLabels() {
		return this->labels;
	}
	float**** getTrainX() {
		return this->trainX;
	}
	float**** getTestX() {
		return this->testX;
	}
	float**** getImgs() {
		return this->imgs;
	}
	float** getTrainY() {
		return this->trainY;
	}
	float** getTestY() {
		return this->testY;
	}
	int getLabelsNum() {
		return this->labelsNum;
	}
	int getTrainNum() {
		return this->trainNum;
	}
	int getTestNum() {
		return this->testNum;
	}
	int getImgsNum() {
		return this->imgsNum;
	}
};