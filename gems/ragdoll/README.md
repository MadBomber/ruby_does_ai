# Ragdoll

Ragdoll is a Rails Engine designed for document ingestion and search. It allows you to import documents, vectorize them, and perform searches using vector representations.

## Installation as a Rails Engine

To use Ragdoll as a Rails Engine, add this line to your application's Gemfile:

```bash
bundle add ragdoll
```

And then execute:

```bash
bundle install
```

Or install it yourself as:

```bash
gem install ragdoll
```

## Usage as a Rails Engine

### Importing Documents

To import documents from a file, glob, or directory, use the following command:

```bash
ragdoll import PATH
```

- `PATH`: The path to the file or directory to import.
- Use the `-r` or `--recursive` option to import files recursively from directories.
- Use the `-j` or `--jobs` option to specify the number of concurrent import jobs.

### Managing Jobs

To manage import jobs, use the following command:

```bash
ragdoll jobs [JOB_ID]
```

- `JOB_ID`: The ID of a specific job to manage.
- Use `--stop`, `--pause`, or `--resume` to control a specific job.
- Use `--stop-all`, `--pause-all`, or `--resume-all` to control all jobs.

### Searching Documents

To search the database with a prompt, use the following command:

```bash
ragdoll search PROMPT
```

- `PROMPT`: The search prompt as a string or use the `-p` option to specify a file containing the prompt text.
- Use the `--max_count` option to specify the maximum number of results to return.
- Use the `--rerank` option to rerank results using keyword search.

## Development and Contribution

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake test` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/[USERNAME]/ragdoll.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
