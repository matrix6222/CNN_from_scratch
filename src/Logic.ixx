#include "asio/ip/tcp.hpp"
#include <iostream>
#include <fstream>

import Config;
import Model;
import Data;

export module Logic;


export class Logic {
private:
	Config config;

	bool modeEval() {
		Model model = Model();
		model.createModelFromVector(this->config.getLayers());
		if (model.compile(this->config.getLearningRate(), this->config.getBatchSize())) {
			if (model.loadWeight(this->config.getWeigthsPath())) {
				Data data = Data(this->config.getImgY(), this->config.getImgX(), this->config.getImgC(), this->config.getLabelsNum());
				if (data.loadImgs(this->config.getParams().getImagesPath()) && data.loadLabels(this->config.getLabelsPath())) {
					std::cout << model.eval(&data) << std::endl;
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
	bool modeServer() {
		Model model = Model();
		model.createModelFromVector(this->config.getLayers());
		if (model.compile(this->config.getLearningRate(), this->config.getBatchSize())) {
			if (model.loadWeight(this->config.getWeigthsPath())) {
				Data data = Data(this->config.getImgY(), this->config.getImgX(), this->config.getImgC(), this->config.getLabelsNum());
				while (true) {
					asio::ip::tcp::endpoint address(asio::ip::address::from_string(this->config.getParams().getIp()), this->config.getParams().getPort());
					asio::io_context io_context;
					asio::ip::tcp::acceptor acceptor(io_context, address);
					asio::ip::tcp::socket socket(io_context);
					acceptor.accept(socket);

					std::array<unsigned char, 65535> buffer;
					size_t len = socket.read_some(asio::buffer(buffer));
					

					if (data.loadImageFromArray(buffer, len) && data.loadLabels(this->config.getLabelsPath())) {
						socket.send(asio::buffer(model.eval(&data)));
						socket.close();
					}
					else {
						socket.send(asio::buffer("Niepoprawny obrazek"));
						socket.close();
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
	bool modeClient() {
		std::ifstream file(this->config.getParams().getImagesPath(), std::ios::binary);
		if (file.is_open()) {
			std::string buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
			file.close();
			if (buffer.size() < 65536) {
				asio::ip::tcp::endpoint address(asio::ip::address::from_string(this->config.getParams().getIp()), this->config.getParams().getPort());
				asio::ip::tcp::iostream stream;
				stream.connect(address);
				if (stream) {
					stream << buffer;
					std::stringstream ss;
					ss << stream.rdbuf();
					stream.close();
					std::cout << ss.str();
					return true;
				}
				else {
					std::cout << "Nie udalo sie polaczyc z hostem" << std::endl;
					return false;
				}
			}
			else {
				std::cout << "Plik za duzy" << std::endl;
				return false;
			}
		}
		else {
			std::cout << "Nie mozna otworzyc pliku" << std::endl;
			return false;
		}
	}
	bool modeLearnNew() {
		Model model = Model();
		model.createModelFromVector(this->config.getLayers());
		if (model.compile(this->config.getLearningRate(), this->config.getBatchSize())) {
			Data data = Data(this->config.getImgY(), this->config.getImgX(), this->config.getImgC(), this->config.getTrainNum(), this->config.getTestNum(), this->config.getLabelsNum());
			if (data.loadTrainX(this->config.getTrainXPath()) && data.loadTrainY(this->config.getTrainYPath()) && data.loadTestX(this->config.getTestXPath()) && data.loadTestY(this->config.getTestYPath())) {
				if (model.train(this->config.getEpochs(), &data, this->config.getWeigthsPath())) {
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
	bool modeLearnContinue() {
		Model model = Model();
		model.createModelFromVector(this->config.getLayers());
		if (model.compile(this->config.getLearningRate(), this->config.getBatchSize())) {
			if (model.loadWeight(this->config.getWeigthsPath())) {
				Data data = Data(this->config.getImgY(), this->config.getImgX(), this->config.getImgC(), this->config.getTrainNum(), this->config.getTestNum(), this->config.getLabelsNum());
				if (data.loadTrainX(this->config.getTrainXPath()) && data.loadTrainY(this->config.getTrainYPath()) && data.loadTestX(this->config.getTestXPath()) && data.loadTestY(this->config.getTestYPath())) {
					if (model.train(this->config.getEpochs(), &data, this->config.getWeigthsPath())) {
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
	Logic() {
	}
	bool run(const int argc, const char* argv[]) {
		Config config = Config();
		if (config.loadConfig(argc, argv)) {
			this->config = config;
			int mode = config.getParams().getMode();
			if (mode == 0) {
				if (this->modeEval()) {
					return true;
				}
				else {
					return false;
				}
			}
			else if (mode == 1) {
				if (this->modeServer()) {
					return true;
				}
				else {
					return false;
				}
			}
			else if (mode == 2) {
				if (this->modeClient()) {
					return true;
				}
				else {
					return false;
				}
			}
			else if (mode == 3) {
				if (this->modeLearnNew()) {
					return true;
				}
				else {
					return false;
				}
			}
			else if (mode == 4) {
				if (this->modeLearnContinue()) {
					return true;
				}
				else {
					return false;
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