[<img align="right" src="https://cdn.buymeacoffee.com/buttons/default-orange.png" width="217px" height="51x">](https://www.buymeacoffee.com/pradolima)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)


# Coleman4HCS


### Solving the Test Case Prioritization using Multi-Armed Bandit Algorithms

**COLEMAN** (_**C**ombinatorial V**O**lati**LE** **M**ulti-Armed B**AN**dit_) is a Multi-Armed Bandit (MAB) based approach designed 
to address the Test Case Prioritization in Continuous Integration (TCPCI) Environments in a cost-effective way.

We designed **COLEMAN** to be generic regarding the programming language in the system under test, 
and adaptive to different contexts and testers' guidelines.

Modeling a MAB-based approach for TCPCI problem gives us some advantages in relation to studies found in the literature, as follows:

- It learns how to incorporate the feedback from the application of the test cases thus incorporating  diversity in the test suite prioritization;
- It uses a policy to deal with the Exploration vs Exploitation (EvE) dilemma, thus mitigating the problem of beginning without knowledge (learning) and adapting to changes in the execution environment, for instance, the fact that some test cases are added (new test cases) and removed (obsolete test cases) from one cycle to another (volatility of test cases);
- It is model-free. The technique is independent of the development environment and programming language, and does not require any analysis in the code level; 
- It is more lightweight, that is, needs only the historical failure data to execute, and has higher performance.

In this way, this repository contains the COLEMAN's implementation.
For more information about **COLEMAN** read **Ref1** in [References](#references).

Furthermore, this repository contains the adaptation to deal with Highly-Configurable System (HCS) context through two strategies:
- **Variant Test Set Strategy** (VTS) that relies on the test set specific for each variant; and 
- **Whole Test Set Strategy** (WST) that prioritizes the test set composed by the union of the test cases of all variants.

For more information about **WTS** and **VTS** read **Ref2** in [References](#references).

![](https://img.shields.io/badge/python-3.6+-blue.svg)


# Getting started

- [Citation](#citation)
- [Installing required dependencies](#installing-required-dependencies)
- [Datasets](#datasets)	
- [About the files input](#about-the-files-input)	
- [Using the tool](#using-the-tool)
	- [Running for a system non HCS](#running-for-a-system-non-hcs)
	- [Running for an HCS system](#running-for-an-hcs-system)
	  - [Whole Test Set Strategy](#whole-test-set-strategy)
	  - [Variant Test Set Strategy](#variant-test-set-strategy)
- [References](#references)
- [Contributors](#Contributors)
----------------------------------


# Citation

If this tool contributes to a project which leads to a scientific publication, I would appreciate a citation.

```
@Article{pradolima2020TSE,
  author  = {Prado Lima, Jackson A. and Vergilio, Silvia R.},
  journal = {IEEE Transactions on Software Engineering},
  title   = {A Multi-Armed Bandit Approach for Test Case Prioritization in Continuous Integration Environments},
  year    = {2020},
  pages   = {12},
  doi     = {10.1109/TSE.2020.2992428},
}

@article{pradolima2021EMSE,
  author  = {Prado Lima, Jackson A. and Mendon{\c{c}}a, Willian D. F. and Vergilio, Silvia R. and Assun{\c{c}}{\~a}o, Wesley K. G.},
  journal = {Empirical Software Engineering},  
  title   = {{Cost-effective learning-based strategies for test case prioritization in Continuous Integration of Highly-Configurable Software}},  
  year    = {2021}
}

```

# Installing required dependencies

The following command allows to install the required dependencies:

```
 $ pip install -r requirements.txt
 ```

# Datasets 

The datasets used in the examples (and much more datasets) are available at [Harvard Dataverse Repository](https://dataverse.harvard.edu/dataverse/gres-ufpr).
You can create your own dataset using out [GitLab CI - Torrent tool](https://github.com/jacksonpradolima/gitlabci-torrent) or our adapted version from [TravisTorrent tool](https://github.com/jacksonpradolima/travistorrent-tools).
Besides that, you can extract relevant information about each system using our tool named [TCPI - Dataset - Utils](https://github.com/jacksonpradolima/tcpci-dataset-utils).

# About the files input

**COLEMAN** considers two kind of *csv files*: **features-engineered** and **data-variants**. 
The second file, **data-variants.csv**, is used by the HCS, and it represents all results from all variants.
The information is organized by commit and variant.

- **features-engineered.csv** contains the following information:
  - **Id**: unique numeric identifier of the test execution; 
  - **Name**: unique numeric identifier of the test case; 
  - **BuildId**: a value uniquely identifying the build; 
  - **Duration**: approximated runtime of the test case;  
  - **LastRun**: previous last execution of the test case as *DateTime*;
  - **Verdict**: test verdict of this test execution (Failed: 1, Passed: 0).

- **data-variants.csv** contains all information that **features-engineered.csv** has, and in addition the following information:
  - **Variant**: variant name.

In this way,  **features-engineered.csv** organize the information for a single system or variant, and 
**data-variants.csv** track the information for all variants used during the software life-cycle (for each commit). 
During the **COLEMAN**'s execution, we use **data-variants** to identify the variants used in a current commit and apply the **WTS** strategy.

#  Using the tool

## Running for a system non HCS

To run COLEMAN, do:

```
python main.py --project_dir 'data'  --policies 'Random' 'FRR' --output_dir 'results/experiments' --datasets 'alibaba@druid' 
```

**where:** 
- `--project_dir` is the directory that contains your system. For instance, we desire to run the algorithm for the systems that are inside the directory **data**. Please, you must to inform the complete path.
- `--datasets` is an array that represents the datasets to analyse. It's the folder name inside `--project_dir` which contains the required file inputs.
- `--output_dir` is the directory where we will save the results.

The another parameters available are:
- `--sched_time_ratio` that represents the Schedule Time Ratio, that is, time constraints that represents the time available to run the tests. **Default**: 0.1 (10%), 0.5 (50%), and 0.8 (80%) of the time available.
- `--rewards` defines the reward functions to be used, available RNFailReward and TimeRankReward (See **Ref1** in [References](#references)). 
- `--parallel_pool_size` is the number of threads to run **COLEMAN** in parallel.
- `--scaling_factor_frr` is an array that contains the scaling factors to be used by the FRRMAB policy
- `--scaling_factor_ucb` is an array that contains the scaling factors to be used by the UCB policy
- `--epsilon` is an array that contains the epsilon values to be used by the Epsilon policy
- `--window_sizes` is an array that contains the sliding window sizes to be used with FRRMAB policy


## Running for an HCS system

In this option, we apply two strategies to find optimal solutions for the variants: **WTS** and **VTS**.
For more information read **Ref2** in [References](#references).

### Whole Test Set Strategy

**WTS** prioritizes the test set composed by the union of the test cases of all variants. To run this strategy, do: 

```
python main.py --project_dir 'data/libssh@libssh-mirror' --considers_variants 'true' --datasets 'libssh@total' --output_dir 'results/optimal_deterministic'
```

**where:** 
- `--project_dir` is the directory that contains your system. To run the variants of the **libssh**, we created subfolders inside the **libssh@libssh-mirror** directory. In this way, **libssh@libssh-mirror** represents the *project name*. Please, you need to inform the complete path.
- `--datasets` is an array that represents the datasets to analyse. It's the folder name inside `--project_dir` which contains the required file inputs. 
- `--considers_variants` is a flag to consider prioritizing the variants of the systems based on the **WTS** strategy. 
- `--output_dir` is the directory where we will save the results.

The another parameters available are:
- `--sched_time_ratio` that represents the Schedule Time Ratio, that is, time constraints that represents the time available to run the tests. **Default**: 0.1 (10%), 0.5 (50%), and 0.8 (80%) of the time available.
- `--rewards` defines the reward functions to be used, available RNFailReward and TimeRankReward (See **Ref1** in [References](#references)). 
- `--parallel_pool_size` is the number of threads to run **COLEMAN** in parallel.
- `--scaling_factor_frr` is an array that contains the scaling factors to be used by the FRRMAB policy
- `--scaling_factor_ucb` is an array that contains the scaling factors to be used by the UCB policy
- `--epsilon` is an array that contains the epsilon values to be used by the Epsilon policy
- `--window_sizes` is an array that contains the sliding window sizes to be used with FRRMAB policy

### Variant Test Set Strategy 

**VTS** prioritizes each variant as a system, that is, treating each variant independently. 
To run this strategy is similar to **WTS**, do: 

```
python main.py --project_dir "data/libssh@libssh-mirror" --datasets 'libssh@CentOS7-openssl' 'libssh@CentOS7-openssl 1.0.x-x86-64' 
```

**where:** each dataset represents a variant.

In this command, we run each variant as single system. 

# References

- 📖 [**Ref1**] [A Multi-Armed Bandit Approach for Test Case Prioritization in Continuous Integration Environments](https://doi.org/10.1109/TSE.2020.2992428) published at **IEEE Transactions on Software Engineering (TSE)**
- 📖 [**Ref2**] [Learning-based prioritization of test cases in continuous integration of highly-configurable software](https://doi.org/10.1145/3382025.3414967) published at **Proceedings of the 24th ACM Conference on Systems and Software Product Line (SPLC'20)**

# Contributors

- 👨‍💻 Jackson Antonio do Prado Lima <a href="mailto:jacksonpradolima@gmail.com">:e-mail:</a>

<a href="https://github.com/jacksonpradolima/coleman4hcs/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jacksonpradolima/coleman4hcs" />
</a>
