#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

using namespace std;

int main(int argc, char** argv)
{
    const char* exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    argv_vector.push_back(const_cast<char*>(exclude));
    for (int i = 1; i < argc; i++) {
        argv_vector.push_back(argv[i]);
    }
    argc++;

    ::testing::InitGoogleTest(&argc, argv_vector.data());
    auto start = std::chrono::system_clock::now();
    int rc = RUN_ALL_TESTS();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - start);
    std::cout << ("[MAIN] Tests finished: Time: %d ms Exit code: %d", elapsed.count(), rc);

    return rc;
}
