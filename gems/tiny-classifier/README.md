# README

[![Build Status](https://travis-ci.org/piroor/tiny-classifier.svg?branch=master)](https://travis-ci.org/piroor/tiny-classifier)

## Name

tiny-classifier

## Description

On-memory text classifier command line tool, based on naive bayes.

In other words, this is a wrapper for the [classifier-reborn](https://github.com/jekyll/classifier-reborn) library, to use it from other non-ruby programs e.g. [tweetbot.sh, a Twitter bot written in Bash script](https://github.com/piroor/tweetbot.sh).

## Install

```
% sudo apt install ruby-dev build-essential
% gem install tiny-classifier
```

If you want to use `--tokenizer=mecab`, you need to install MeCab like:

```
% sudo apt install mecab mecab-ipadic-utf8
```

This is example on Ubuntu.

## Basic usage

Training:

```
% echo "Hello, world!"        | tc-train --categories=positive,negative positive
% echo "I'm very very happy!" | tc-train --categories=positive,negative positive
% echo "I'm so bad..."        | tc-train --categories=positive,negative negative
% echo "Oh my god!"           | tc-train --categories=positive,negative negative
```

The training data will be saved as `tc.negative-positive.dat` (`tc.` is the fixed prefix, `.dat` is the fixed suffix. The middle part is filled by given categories automatically.) in the current directory. If you hope the file to be saved in any different place, please specify `--base-dir=/path/to/data/directory`.

Untraining for mistakes:

```
% echo "I'm so bad..." | tc-untrain --categories=positive,negative positive
```

Testing to classify:

~~~
% echo "Happy day?" | tc-classify --categories=positive,negative
positive
~~~

If you think that the classifier has been enoughly trained, then you can generate a fixed classifier:

~~~
% tc-generate-classifier --categories=positive,negative --output-dir=/path/to/dir
~~~

Then a fixed classifier (executable Ruby script) will be generated as `tc-classify-negative-positive` (`tc-classify-` is the fixed prefix, rest is filled by given categories automatically.)

~~~
% ls /path/to/dir/
tc-classify-negative-positive
% echo "Happy day?" | /path/to/dir/tc-classify-negative-positive
positive
~~~

Note: the generated classifier script is not distributable - it strongly depend on this `tiny-classifier` gem itself. If you need to deploy it to other computers, you also need to install `tiny-classifier` gem even if you never run `tc-train` or other training commands on the computer.

## Command line parameters

### Common

`-c`, `--categories=CATEGORIES` (required)
:  A comman-separated list of categories. You should use only alphabetic characters. (Non-alphabetical characters will cause problems.)

`-d`, `--data-dir=PATH` (optional)
: The path to the directory that the training data to be saved. The current directory is the default value.

`-t`, `--tokenizer=TOKENIZER` (optional)
: Tokenizer for input which is not separated by whitespaces. Possible values are: only `mecab`.

### `tc-train` and `tc-untrain` specific parameters

Both `tc-train` and `tc-untrain` require one command line argument: the category. You need to specify one of categories given via the `--categories` parameter.

### `tc-generate-classifier` specific parameters

`-o`, `--output-dir=PATH` (optional)
: The path to the directory that the classifier to be saved. The current directory is the default value.

## Copyright

Copyright (c) 2017 YUKI "Piro" Hiroshi

## License

GPLv3 or later. See LICENSE.txt for details.
