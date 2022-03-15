## How to use documentation markup features

![Lang switching tabs](images/lang-tabs.png)

1. To add code for different language variants, use blocks of code in the markdown, one after the other, indicating the language:
```
    ``` bash
        # here your code
    ```
    
    ``` python
        # here your code
    ```
```
2. The copy button appears by itself on any outline code blocks.
3. To emphasize important information, use `<aside role="status">your text here</aside>` right in markdown

![An element with important information](images/info.png)

1. To emphasize dangerous actions or warn users, use `<aside role=”alert”>your text here</aside>` right in markdown

![An element with warning](images/warning.png)
