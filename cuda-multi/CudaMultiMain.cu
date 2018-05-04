
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include "Version1.cuh"

using namespace std;
using namespace trinom;

void computeAllOptions(const Args &args)
{
    if (args.test)
    {
        cout << "Cuda multiple options per thread block version " << args.version << endl;
    }
    
    int yieldSize;
    vector<OptionConstants> optionConstants;
    cuda::init(args, optionConstants, yieldSize);

    vector<real> results;
    results.resize(optionConstants.size());

    switch (args.version)
    {
        case 1:
            cuda::computeOptionsNaive(optionConstants, yieldSize, results, args.test);
            break;
    }

    if (!args.test)
    {
        Arrays::write_array(cout, results);
    }
}

int main(int argc, char *argv[])
{
    auto args = Args::parseArgs(argc, argv);

    computeAllOptions(args);

    return 0;
}
