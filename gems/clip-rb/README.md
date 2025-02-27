# clip-rb

[![Gem Version](https://badge.fury.io/rb/clip-rb.svg)](https://badge.fury.io/rb/clip-rb)  
[![Test](https://github.com/khasinski/clip-rb/workflows/clip-rb/badge.svg)](https://github.com/khasinski/clip-rb/actions/workflows/main.yml)

**clip-rb** is a Ruby implementation of [OpenAI CLIP](https://openai.com/index/clip/) powered by ONNX models—no Python required!

CLIP (Contrastive Language–Image Pre-training) is a powerful neural network developed by OpenAI. It connects text and images by learning shared representations, enabling tasks such as image-to-text matching, zero-shot classification, and visual search. With clip-rb, you can easily encode text and images into high-dimensional embeddings for similarity comparison or use in downstream applications like caption generation and vector search.

## Why do I need this?

It's a key piece of tech to write an unlabeled image search. You can upload images and then search them by text or using another image as a reference.

The other thing you need is a knn search in a vector database. Generate embeddings for images using this, store them in the database. When user wants to search generate embeddings for their query or image, and do a vector search to find the relevant images.

See [neighbor gem](https://github.com/ankane/neighbor) to learn more about vector search.

---

## Requirements

- Ruby 3.0.0 or later
- ONNX CLIP models (downloaded automatically on first use)
- XLM Roberta CLIP model (for multilingual support)

---

## Installation

Add the gem to your application by executing:

```bash
bundle add clip-rb
```

If bundler is not being used to manage dependencies, install the gem by executing:

```bash
gem install clip-rb
```

## Usage

```ruby
require 'clip'

# This will download the models on first use (default path is .clip_models)
# If you don't want this behavior you can pass the path to the models as an argument.
clip = Clip::Model.new 

text_embedding = clip.encode_text("a photo of a cat")
# => [0.15546110272407532, 0.07329428941011429, ...]

image_embedding = clip.encode_image("test/fixtures/test.jpg")
# => [0.22115306556224823, 0.19343754649162292, ...]
```

💡 Tip: Use cosine similarity for KNN vector search when comparing embeddings!

## Multilingual text embeddings

Since the original CLIP only supports English embeddings this gem now has added support for multilingual text embeddings using the XLM Roberta model.

```ruby
require 'clip'

# This will download the models on first use (default path is .clip_models/multilingual)
# If you don't want this behavior you can pass the path to the models as an argument.
clip = Clip::MultilingualModel.new

text_embedding = clip.encode_text("un photo de un gato")
# => [0.15546110272407532, 0.07329428941011429, ...]

image_embedding = clip.encode_image("test/fixtures/test.jpg")
# => [0.22115306556224823, 0.19343754649162292, ...]
```

## CLI

Additionally you can fetch embeddings by calling:

```bash
$ clip-embed-text "a photo of a cat"
$ clip-embed-image test/fixtures/test.jpg
```

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/khasinski/clip-rb. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/[USERNAME]/clip-rb/blob/main/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the clip-rb project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/[USERNAME]/clip-rb/blob/main/CODE_OF_CONDUCT.md).
