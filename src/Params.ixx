#include <iostream>
#include <regex>

export module Params;


export class Params {
private:
	int mode = -1;
	int port = -1;
	std::string ip;
	std::string configPath;
	std::string imagesPath;
public:
	Params() {
	}
	bool loadParams(const int& argc, const char* argv[]) {
		std::regex ipPortRegex("^(([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\\.){3}([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5]):([0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$");
		bool flags[4] = { 0,0,0,0 };
		std::string arg;
		std::string val;
		if (argc == 7 || argc == 5) {
			for (int x = 1; x < argc; x = x + 2) {
				arg = std::string(argv[x]);
				val = std::string(argv[x + 1]);
				if (arg == "-t" && flags[0] == 0) {
					flags[0] = 1;
					if (val == "eval") {
						this->mode = 0;
					}
					else if (val == "server") {
						this->mode = 1;
					}
					else if (val == "client") {
						this->mode = 2;
					}
					else if (val == "learnNew") {
						this->mode = 3;
					}
					else if (val == "learnContinue") {
						this->mode = 4;
					}
					else {
						std::cout << "Nieprawidlowy tryb" << std::endl;
						return false;
					}
				}
				else if (arg == "-c" && flags[1] == 0) {
					flags[1] = 1;
					this->configPath = val;
				}
				else if (arg == "-p" && flags[2] == 0) {
					flags[2] = 1;
					if (std::regex_match(val, ipPortRegex)) {
						this->port = stoi(val.substr(val.find(":") + 1));
						this->ip = val.substr(0, val.find(":"));
					}
					else {
						std::cout << "Nieprawidlowy adres" << std::endl;
						return false;
					}
				}
				else if (arg == "-i" && flags[3] == 0) {
					flags[3] = 1;
					this->imagesPath = val;
				}
				else {
					std::cout << "Nieprawidlowy przelacznik" << std::endl;
					return false;
				}
			}
		}
		else {
			std::cout << "Nieprawidlowe przelaczniki" << std::endl;
			return false;
		}
		if (this->mode == 0) {
			if (flags[0] == 1 && flags[1] == 1 && flags[2] == 0 && flags[3] == 1) {
				return true;
			}
			else {
				std::cout << "Nieprawidlowe przelaczniki" << std::endl;
				return false;
			}
		}
		else if (this->mode == 1) {
			if (flags[0] == 1 && flags[1] == 1 && flags[2] == 1 && flags[3] == 0) {
				return true;
			}
			else {
				std::cout << "Nieprawidlowe przelaczniki" << std::endl;
				return false;
			}
		}
		else if (this->mode == 2) {
			if (flags[0] == 1 && flags[1] == 0 && flags[2] == 1 && flags[3] == 1) {
				return true;
			}
			else {
				std::cout << "Nieprawidlowe przelaczniki" << std::endl;
				return false;
			}
		}
		else if (this->mode == 3) {
			if (flags[0] == 1 && flags[1] == 1 && flags[2] == 0 && flags[3] == 0) {
				return true;
			}
			else {
				std::cout << "Nieprawidlowe przelaczniki" << std::endl;
				return false;
			}
		}
		else if (this->mode == 4) {
			if (flags[0] == 1 && flags[1] == 1 && flags[2] == 0 && flags[3] == 0) {
				return true;
			}
			else {
				std::cout << "Nieprawidlowe przelaczniki" << std::endl;
				return false;
			}
		}
		else {
			std::cout << "Nieprawidlowe przelaczniki" << std::endl;
			return false;
		}
	}

	int getMode() {
		return this->mode;
	}
	int getPort() {
		return this->port;
	}
	std::string getIp() {
		return this->ip;
	}
	std::string getConfigPath() {
		return this->configPath;
	}
	std::string getImagesPath() {
		return this->imagesPath;
	}
};