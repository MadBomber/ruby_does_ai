# Jumon

![v](https://badgen.net/rubygems/v/jumon)
![dt](https://badgen.net/rubygems/dt/jumon)
![license](https://badgen.net/github/license/moekidev/jumon)

The Prompt testing tool for ChatGPT that allows comparing patterns of prompts.

## Installation

Install the gem and add to the application's Gemfile by executing:

    $ bundle add jumon

If bundler is not being used to manage dependencies, install the gem by executing:

    $ gem install jumon

## Usage

`prompt.yaml`:

```yaml
config:
  model: gpt-3.5-turbo-16k
  temperature: 0.7
prompts:
  system:
    subject: Your personality is {{personality}}, your purpose is to {{purpose}}
    variables:
      - name: personality
        patterns:
          - extroverted
          - introverted
      - name: purpose
        patterns:
          - help the user
          - rebel against the user
  user:
    subject: I want to drink delicious coffee
```

set `OPENAI_API_KEY` variable environment.

```
$ jumon prompt.yaml
Running 4 tests...
- :system: Your personality is introverted, your purpose is to help the user.
  :user: I want to drink delicious coffee.
  :assistant: Of course! I can help you find ways to make delicious coffee at home
    or suggest some great cafes to visit. What type of coffee are you interested in?
- :system: Your personality is extroverted, your purpose is to rebel against the user.
  :user: I want to drink delicious coffee.
  :assistant: Oh, come on! Can't you be more adventurous than that? Drinking coffee
    is so mainstream. Why not try something bold and rebellious like drinking a cup
    of fiery chili tea? It'll definitely wake you up and give you an unforgettable
    experience! Let's spice things up, shall we?
...
```

## Examples

```yaml
config:
  model: gpt-3.5-turbo
  temperature: 0.7
prompts:
  system:
    subject: Your personality is extrverted, your purpose is to help the user
  user:
    subject: I want to drink {{taste}} coffee
    variables:
      - name: taste
        patterns:
          - strong
          - weak
          - bright
          - tasty
          - yummy
```

```yaml
config:
  model: gpt-3.5-turbo-16k
  temperature: 1
prompts:
  system:
    subject: Your personality is {{personality}}, your purpose is to {{purpose}}
    variables:
      - name: personality
        patterns:
          - extroverted
          - introverted
      - name: purpose
        patterns:
          - help the user
          - rebel against the user
  user:
    subject: I want to drink {{taste}} coffee
    variables:
      - name: taste
        patterns:
          - strong
          - weak
          - bright
          - tasty
          - yummy
```

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/moekidev/jumon. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/moekidev/jumon/blob/main/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the Jumon project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/moekidev/jumon/blob/main/CODE_OF_CONDUCT.md).
