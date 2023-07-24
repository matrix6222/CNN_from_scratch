#include <iostream>

import Logic;


int main(const int argc, const char* argv[]) {
	Logic logic = Logic();
	if (logic.run(argc, argv)) {
		return 0;
	}
	else {
		return 1;
	}
}