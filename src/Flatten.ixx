export import Layer;

export module Flatten;


export class Flatten : public Layer {
private:
	int batchSize = -1;
	int outputX = -1;

	Layer* prev = nullptr;

	float** outputArray = nullptr;
	float** dB = nullptr;
public:
	Flatten() {
	}
	~Flatten() {
		for (int bs = 0; bs < batchSize; bs++) {
			delete[] this->outputArray[bs];
			delete[] this->dB[bs];
		}
		delete[] this->outputArray;
		delete[] this->dB;
	}
	bool compile(Layer* prev, float learningRate, int batchSize) {
		if (prev->getOutputDimsCount() == 3) {
			this->prev = prev;
			this->outputX = prev->getOutputY() * prev->getOutputX() * prev->getOutputC();
			this->batchSize = batchSize;

			this->outputArray = new float* [batchSize];
			for (int bs = 0; bs < batchSize; bs++) {
				this->outputArray[bs] = new float[this->outputX];
			}

			this->dB = new float* [batchSize];
			for (int bs = 0; bs < batchSize; bs++) {
				this->dB[bs] = new float[this->outputX];
			}
			return true;
		}
		else {
			return false;
		}
	}
	int getOutputDimsCount() {
		return 1;
	}
	int getOutputX() {
		return this->outputX;
	}
	float** getOutputArray1D() {
		return this->outputArray;
	}

	void forward(int stepNum) {
		int prevOutputY = prev->getOutputY();
		int prevOutputX = prev->getOutputX();
		int prevOutputC = prev->getOutputC();
		float**** prevOutputArray = prev->getOutputArray3D();
		for (int y = 0; y < prevOutputY; y++) {
			for (int x = 0; x < prevOutputX; x++) {
				for (int c = 0; c < prevOutputC; c++) {
					this->outputArray[stepNum][y * prevOutputX * prevOutputC + x * prevOutputC + c] = prevOutputArray[stepNum][y][x][c];
				}
			}
		}
	}

	void calcPrevdB(int stepNum) {
		float**** prevdB = this->prev->getdB3D();
		int prevOutputY = this->prev->getOutputY();
		int prevOutputX = this->prev->getOutputX();
		int prevOutputC = this->prev->getOutputC();
		for (int y = 0; y < prevOutputY; y++) {
			for (int x = 0; x < prevOutputX; x++) {
				for (int c = 0; c < prevOutputC; c++) {
					prevdB[stepNum][y][x][c] = this->dB[stepNum][y * prevOutputX * prevOutputC + x * prevOutputC + c];
				}
			}
		}
	}
	float** getdB1D() {
		return dB;
	}
};