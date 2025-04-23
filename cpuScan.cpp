// cpu_scan.cpp
#include <vector>
#include <iostream>

void cpu_exclusive_scan(const std::vector<int>& input, std::vector<int>& output) {
    output[0] = 0;
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

int main() {
    std::vector<int> input = {3, 1, 7, 0, 4, 1, 6, 3};
    std::vector<int> output(input.size());

    cpu_exclusive_scan(input, output);

    std::cout << "Input: ";
    for (int x : input) std::cout << x << " ";
    std::cout << "\nOutput (Exclusive Scan): ";
    for (int x : output) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
