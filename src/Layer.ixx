#include "json.hpp"
#include <vector>

export module Layer;


export class Layer {
public:
	virtual bool compile(Layer* prev, float learningRate, int batchSize) {
		return false;
	}

	virtual bool compile(int batchSize) {
		return false;
	}
	virtual int getOutputDimsCount() {
		return -1;
	}
	virtual float**** getOutputArray3D() {
		return nullptr;
	}
	virtual float** getOutputArray1D() {
		return nullptr;
	}
	virtual int getOutputY() {
		return -1;
	}
	virtual int getOutputX() {
		return -1;
	}
	virtual int getOutputC() {
		return -1;
	}
	
	virtual void forward(float*** input, int stepNum) {
	}
	virtual void forward(int stepNum) {
	}

	virtual void updateWeights(int batchSize) {
	}
	virtual void calcPrevdB(int stepNum) {
	}
	virtual void calcdW(int stepNum) {
	}
	virtual float**** getdB3D() {
		return nullptr;
	}
	virtual void setdWToZero() {
	}
	virtual float** getdB1D() {
		return nullptr;
	}
	
	
	virtual bool setWeight(nlohmann::json weight, nlohmann::json bias) {
		return true;
	}
	virtual nlohmann::json getWeight() {
		nlohmann::json j;
		return j;
	}
	virtual int getWeightCount() {
		return 0;
	}

	float sigmoid(float x) {
		return 1.0f / (1.0f + exp(-x));
	}
};