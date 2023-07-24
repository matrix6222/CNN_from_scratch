export import Layer;

export module Input;


export class Input : public Layer {
private:
	int batchSize = -1;
	int outputY = -1;
	int outputX = -1;
	int outputC = -1;

	float**** outputArray = nullptr;
public:
	Input(int Y, int X, int C) {
		this->outputY = Y;
		this->outputX = X;
		this->outputC = C;
	}
	~Input() {
		for (int bs = 0; bs < batchSize; bs++) {
			for (int y = 0; y < this->outputY; y++) {
				for (int x = 0; x < this->outputX; x++) {
					delete[] this->outputArray[bs][y][x];
				}
				delete[] this->outputArray[bs][y];
			}
			delete[] this->outputArray[bs];
		}
		delete[] this->outputArray;
	}
	bool compile(int batchSize) {
		this->batchSize = batchSize;

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
		return true;
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

	void forward(float*** input, int stepNum) {
		for (int y = 0; y < this->outputY; y++) {
			for (int x = 0; x < this->outputX; x++) {
				for (int c = 0; c < this->outputC; c++) {
					this->outputArray[stepNum][y][x][c] = input[y][x][c];
				}
			}
		}
	}
};