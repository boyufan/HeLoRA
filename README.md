# HeteroLoRA

## Task 1: custom the load_dataset() function
* can load different datasets in Huggingface Datasets
* can set different iid/non-iid settings, including dirchlet

## Task 2: custom the aggregation strategy
* define a new strategy extended from flower.server.strategy
* overwrite the function aggregate_fit() to achieve heterogeneous model aggregation
* implement the basic HeteroLoRA algorithm, where different parameter shapes are aggregated by paadding
* adjust other functions to obtain the heterogeneous parameters, etc.

For more information, please refer to https://flower.ai/docs/framework/how-to-implement-strategies.html#

Feel free to adjust the parameters and define new helper functions

Estimated Completion Time: before 2024/03/01
