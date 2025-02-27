# AICommit
Use your own spell against you.

## Setup
### Prerequisites
- Ruby 2.6 or later
- An OpenAI API key

### Installation
Install the aicommit gem:
```bash
gem install aicommit
```
### Upgrading
To upgrade to the latest version of AICommit, run:

```bash
gem update aicommit
```

### Usage
#### Generate a commit message
To generate a commit message based on the changes in your Git repository:

1. Run the following command at the root of your project:
    ```bash
    aicommit
    ```

2. The AI model will generate a commit message based on the changes in your Git repository.

3. Review the generated commit message.

4. To commit the staged changes with the generated commit message, enter Y at the prompt.
To regenerate a new commit message, enter R.
To enter a new commit message by yourself, enter N.
To quit without committing, enter Q.

#### Set OpenAI API key

To set your OpenAI API key manually, run the following command:

```bash
aicommit --config
```
Get your API key from [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys)


#### Show version

To show the version of AICommit, run the following command:

```bash
aicommit --version
```

### How it works
AICommit uses OpenAI's GPT-3.5 AI model to generate commit messages based on the changes in your Git repository.

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/jackal998/aicommit/blob/95ca693e3cf4c87dcd4916aadb2459efea0823ae/LICENSE) file for details.
