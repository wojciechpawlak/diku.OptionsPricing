#ifndef ARGS_HPP
#define ARGS_HPP

//#define USE_GETOPT_ARGS

#ifdef USE_GETOPT_ARGS
#include "../common/getoptpp/getopt_pp_standalone.h"
#else
#include "../common/cxxopts/cxxopts.hpp"
#endif

#include "ValuationConstants.hpp"

namespace trinom
{

struct Args
{
    std::string valuations;
    std::vector<SortType> sorts;
    std::vector<int> blockSizes;
    std::vector<int> versions;
    int device;
    int runs;
    bool test;

    Args() {}

    Args(int argc, char *argv[])
    {
        std::vector<std::string> sortOpts;
#ifdef USE_GETOPT_ARGS
        GetOpt::GetOpt_pp cmd(argc, argv);

        // Defaults for single arguments
        runs = 0;
        device = 0;

        cmd >> GetOpt::Option('o', "valuations", valuations);
        cmd >> GetOpt::Option('s', "sort", sortOpts);
        cmd >> GetOpt::Option('v', "version", versions);
        cmd >> GetOpt::Option('r', "runs", runs);
        cmd >> GetOpt::Option('b', "block", blockSizes);
        cmd >> GetOpt::Option('d', "device", device);
        cmd >> GetOpt::OptionPresent('t', "test", test);

#else
        cxxopts::Options options(argv[0], " - example command line options");
        options
            .positional_help("[optional args]")
            .show_positional_help();

        options
            .add_options()
            ("h,help", "Print help")
            ("o,valuations", "Input file", cxxopts::value<std::string>(valuations))
            ("s,sort", "Sorting type applied to valuations", cxxopts::value<std::vector<std::string>>(sortOpts))
            ("b,block", "Block size", cxxopts::value<std::vector<int>>(blockSizes))
            ("v,version", "Version of the kernel", cxxopts::value<std::vector<int>>(versions))
            ("d,device", "GPU Device number", cxxopts::value<int>(device)->default_value("0"))
            ("r,runs", "Number of runs", cxxopts::value<int>(runs)->default_value("0"))
            ("t,test", "Test", cxxopts::value<bool>(test)->default_value("false"))
            ;

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help({ "", "Group" }) << std::endl;
            exit(0);
        }
#endif
        for (auto &sort : sortOpts)
        {
            if (sort.length() == 1)
            {
                auto sortType = (SortType)sort[0];
                if (sortType == SortType::HEIGHT_ASC || sortType == SortType::HEIGHT_DESC || sortType == SortType::WIDTH_ASC //
                    || sortType == SortType::WIDTH_DESC || sortType == SortType::NONE)
                {
                    sorts.push_back(sortType);
                }
            }
        }

        // Defaults for multiple arguments
        if (sorts.empty())
            sorts.push_back(SortType::NONE);
        if (blockSizes.empty())
            blockSizes.push_back(-1);
        if (versions.empty())
            versions.push_back(1);
    }

private:
    std::vector<std::string> sortVals;
};
} // namespace trinom

#endif
