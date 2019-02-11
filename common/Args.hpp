#ifndef ARGS_HPP
#define ARGS_HPP

#include "cxxopts/cxxopts.hpp"
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
        cxxopts::Options options(argv[0], " - example command line options");
        options
            .positional_help("[optional args]")
            .show_positional_help();

        options
            .add_options()
            ("h,help", "Print help")
            ("o,valuations", "Input file", cxxopts::value<std::string>(valuations))
            ("s,sort", "Sorting type applied to valuations", cxxopts::value<std::vector<std::string>>(sortVals))
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

        for (auto &sort : sortVals)
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
