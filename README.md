# jcat-eval

## What is it?

This the jcatGPT testing harness! It is tasked with evaluating jcatGPT's underlying LLM with a set of test cases produced by the JCAT community. The testing process is shown in the figure below:

![image](doc/flow.png)

As such it is not satisifed if the generated code solely compiles - it requires the code to run and satisfy a set of assertions as well. 

## How to install?

```sh
python3 -m pip install .
```

## Run the test
To run the test you can use the following script. It will trigger the testing process, running the test specificed in [tests/test_jcat.py]

```sh
./test.sh
```

## Directory structure
```bash
├── README.md
├── jcat
│   ├── __init__.py
│   ├── java_sandbox.py             # A sandbox which govers the execution of java based tests (supports org.junit and testng)
│   ├── jcat_endpoint.py            # A wrapper to the JCAT REST API
│   ├── jcat_promtps.py             # A mechanism to set the prompt for the JCAT REST API
├── jcat_tests                      # contains the tests produced by the JCAT community
│   ├── generated_deepseekcoder     # contains the responses generated using DeepSeekCoder
│   ├── generated_llama             # contains the responses generated using Code Llama
├── libs                            # contains org.junit and testng java libraries
├── notebooks                       # notebooks to evaluate/visualiase evaluation results 
│   └── analysis.ipynb
├── tests
│   └── test_jcat.py                # main testing function to run all tests
```# radiogunet
