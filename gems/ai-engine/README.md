# AI::Engine

AI::Engine is the fastest way to get AI Assistants into your Rails app! It's a Rails Engine which sets up everything you need to start streaming from OpenAI Assistants in your Rails app.

Full docs can be found at [RailsAI.com](https://railsai.com/docs/installation).

A demo app can be found [here](https://github.com/alexrudall/ai-engine-starter-kit/).

## Usage

You can add AI::Engine to your Gemfile like this:

```
gem "ai-engine", "~> 0.3.0"
```

You then need to add the migrations for the gem:

```
bundle exec rails ai_engine:install:migrations
```

And run them:

```
bundle exec rails db:migrate
```

Full docs can be found at [RailsAI.com](https://railsai.com/docs/installation).

## Engine Development

### Test local version in a Rails app

```bash
gem "ai-engine", path: "../ai-engine"
```

### ENV

The dummy app needs a .env file in the root of the engine for manual and RSpec testing - see .env.example.

### Dummy app

Run the dummy app from the root of the project with `bin/dev` in one tab and `bin/rails s` in another (so debugger will work).

### Tests

Run the tests from the root of the project with `rspec`.

### VCR

AI::Engine uses VCR to record HTTP requests and responses. By default, specs are run against recorded 'cassette' fixtures.

Set OPENAI_ACCESS_TOKEN= in your .env file to run the specs against a live API and re-record all cassettes - this will cost you money!

### Release

First run the specs without VCR so they actually hit the API. This will cost 2 cents or more. Set OPENAI_ACCESS_TOKEN in your environment or pass it in like this:

```
OPENAI_ACCESS_TOKEN=123abc bundle exec rspec
```

Then update the version number in `version.rb`, update `CHANGELOG.md`, run `bundle install` to update Gemfile.lock, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).
