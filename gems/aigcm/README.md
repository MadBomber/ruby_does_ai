# aigcm

The **AI git commit message** generator.

See the [change log](./CHANGELOG.md) for recent changes.

<!-- Tocer[start]: Auto-generated, don't remove. -->

## Table of Contents

  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Options](#options)
    - [Examples](#examples)
  - [Advanced Usage](#advanced-usage)
    - [Defaults](#defaults)
         - [Model](#model)
         - [Style Guide](#style-guide)
    - [Private Repos](#private-repos)
         - [Private Repo Check Limitation](#private-repo-check-limitation)
    - [Style Guide Example](#style-guide-example)
  - [API Key Configuration](#api-key-configuration)
  - [Last Thoughts](#last-thoughts)
    - [Support Custom Workflows](#support-custom-workflows)
    - [Non-deterministic Nature of LLMs](#non-deterministic-nature-of-llms)
  - [Development](#development)
  - [Contributing](#contributing)
  - [License](#license)

<!-- Tocer[finish]: Auto-generated, don't remove. -->


## Overview

**aigcm** is a Ruby gem designed to generate high-quality commit messages for git diffs. It leverages AI to analyze changes in your codebase and create concise, meaningful commit messages following best practices.

## Features

- **Automatic Commit Message Generation**: Automatically generate commit messages based on code diffs.
- **Security-Aware Provider Selection**: Detects execution in a private repository and ensures non-local providers are not used for security reasons. This means that if you are working within a private repository, the gem will default to local providers unless explicitly forced otherwise.
- **Configurable Style Guide**: Allows using a specific style guide for commit messages, either from a default location or specified by the user.
- **AI Model Integration**: Integration with various AI models for enhanced message generation.

## Installation

`aigcm` is a CLI tool deployed as a gem.

```shell
gem install aigcm
```

## Usage

To generate a commit message:

```shell
aigcm [options] [ref]
```

### Options

- `-h, --help`: Display options available.
- `-a, --amend`: Amend the last commit.
- `-c, --context=CONTEXT`: Extra context beyond the diff.
- `-d, --dry`: Dry run the command without making any changes.
- `-m, --model=MODEL`: Specify the AI model to use.
- `--provider=PROVIDER`: Specify the provider (ollama, openai, anthropic, etc). Note: This only needs to be used when the specified model is available from multiple providers; otherwise, the owner of the model is used by default.
- `--force-external`: Force using external AI provider even for private repos.
- `-s, --style=STYLE`: Path to the style guide file. If not provided, the system looks for `COMMITS.md` in the repo root or uses the default style guide.
- `--default`: Print the default style guide and exit the application.
- `--version`: Show version.

### Examples

If your commit involves refactoring a function to improve its performance, you might provide context like:
   ```shell
   aigcm -m MODEL -c "Refactored to improve performance by using algorithm X"
   ```

   This context helps the AI craft a more informative commit message.
  
When your commit is related to a specific JIRA ticket:
   ```shell
   aigcm -m MODEL -c "Resolved issues as per JIRA ticket JIRA-1234"
   ```

Including the JIRA ticket helps relate the commit to external tracking systems.

Including multiple context strings:
   ```shell
   aigcm -m MODEL -c "Refactored for performance" -c "JIRA-1234"
   ```

Multiple context strings can be added by repeating the `-c` option.

Using environment variables in context:
   ```shell
   aigcm -c "Put the work ticket as the first entry on the subject line" -c "Ticket: $TICKET"
   ```

This allows you to dynamically include the value of environment variables in your commit message.

## Advanced Usage

### Defaults

There are two important defaults of which you should be aware.  The model controls how much it costs for aigcm to write the commit message for you.  The style guide impacts how well the commit message is written.

#### Model

If you do not specify a model `aigcm` attempts to use `gpt-4o-mini` from OpenAI.  If you want to use a different model or different provider the best way to do that is to create a shell alias like `alias aigcm='\aigcm --model o3-mini'` or `aigcm='\aigcm --model deepseek-r1 --provider ollama`

#### Style Guide

If you do not have a style guide named `COMMITS.md` in the root directory of your repo's working directory AND you do not specify a path to a style guid file using the `--style` option THEN `aigcm` will use its own default style guide.  To print to STDOUT the default style guide use the `--default` option. 

### Private Repos

When you specify a model or provider which is not associated with localhost processing, this gem issue an ERROR message when it is operating in a private repository.  There are lots of security officiers and bosses who just do not want even a `git diff` to be shared outside of their control regardless of whether the provider "crosses their heart and hopes to die" that they do not use the data you send them.

You can by-pass this check by using the option `--force-external` or my simply using a model/provider combination that you process locally on your computer.

#### Private Repo Check Limitation

This gem uses the `gh` command to determine if the repo is private.  This means 1) your repo needs to be hosted on Github; and 2) you need to have the `gh` command installed on your machine AND logged into Github.  If any one of these conditions is not true, then the gem may terminate with an exception.

### Style Guide Example

The style guide is used as part of the generative AI prompt that instructs the large language model (LLM) how to craft its summary of the `git diff` results.  The see the default style guide use the `--default` option.

You can create your own style guide named `COMMITS.md` in the root directory of your repository.  You can also use the `--style` option to point `aigcm` to your style guide if you choose to keep it in a different place.  This is handy when you want to have consistent commit messages across several different projects.

This would be a simple style guide:

```
- Use conventional commits format (type: description)
- Keep first line under 72 characters
- Use present tense ("add" not "added")
- Be descriptive but concise
- Have fun. Be creative. Add ASCII art if you feel like it.
```

## API Key Configuration

Bring your own API keys to the LLM providers.  This gem uses the generic `ai_client` gem to access LLM APIs.  If you are using a model processed on your localhost, you most likely will not need an API key.  If however, you are using an external provider then you need to acquire an API key from that provider.

The API key should be set in a system environment variable (envar) that has the pattern <provider>_API_KEY.

The following table shows a small sample of the providers supported, the envar required to be set and a link to the place to acquire an API key for the provider.

| Provider Link | envar Name |
| --- | --- |
| [Anthropic](https://www.anthropic.com/) | ANTHROPIC_API_KEY |
| [OpenAI](https://platform.openai.com/signup) | OPENAI_API_KEY |
| [OpenRouter](https://openrouter.ai/) | OPENROUTER_API_KEY |
| etc ... | ?_API_KEY |

Note that OpenRouter supports lots of models from many providers.

## Last Thoughts

### Support Custom Workflows

This gem saves its commit message in the file `.aigcm_msg` at the root directory of the repository.  Its there even if you do a `--dry` run.  This could be handy if you want to incorporate `aigcm` into some larger workflow.

Remember that the style guide can be extended using one or more `--context` strings.  For example you could create a shell alias like this:

```shell
alias gc='aigcm -c "JIRA $JIRA_TICKET"'
```

### Non-deterministic Nature of LLMs

Given the same input an LLM may not (most likely will not) give you the same output as it did before.  This is why we like them.  This is also why we hate them.

When you do a dry run against a set of staged changes using the `--dry` option `aigcm` will print to STDOUT a commit message.  It will also save that same commit message in the file `.aigcm_msg` in the root of your working directory.  The commit will not be made because its a "dry run."

If you wait longer than 1 minute to run `aigcm` without the `--dry` option it will generate a new commit message which likely will be different than the one you read during the dry run.  However, if you run again without the dry run option in less than a minute `aigcm` will use its previous commit message on the assumption that you liked it.

Don't forget that if you make a commit and then decide that you need to change the commit message there is always the `--amend` option available.

## Development

After checking out the repo, run `rake test` to run the tests.  Make sure that all of the tests are passing.

To install this gem onto your local machine, run `bundle exec rake install`.

## Contributing

1. Fork it (<https://github.com/your_username/aigcm/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## License

The gem is available as open-source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
