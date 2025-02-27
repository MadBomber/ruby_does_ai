# Cohere AI

A Ruby gem for interacting with [Cohere AI](https://cohere.com).

![The logo depicts a ruby with a liquid-like interior on a peach background. The gem has a flowing lava appearance, with swirls of red and pink suggesting movement, resembling molten rock. Darker red edges and a circular dark border frame this vibrant, fluid core.](https://raw.githubusercontent.com/gbaptista/assets/main/cohere-ai/cohere-ai-canvas.png)

> _This Gem is designed to provide low-level access to Cohere AI, enabling people to build abstractions on top of it. If you are interested in more high-level abstractions or more user-friendly tools, you may want to consider [Nano Bots](https://github.com/icebaker/ruby-nano-bots) 💎 🤖._

## TL;DR and Quick Start

```ruby
gem 'cohere-ai', '~> 1.1.0'
```

```ruby
require 'cohere-ai'

client = Cohere.new(
  credentials: { api_key: ENV['COHERE_API_KEY'] },
  options: { server_sent_events: true }
)

result = client.chat(
  { model: 'command', message: 'Hi!' }
)
```

Result:
```ruby
{ 'response_id' => '59aac3ec-74c8-4b65-a80c-d4c1916d2511',
  'text' =>
  "Hi there! I'm Coral, an AI-assistant chatbot ready to help you with whatever you need. Is there anything I can assist you with today? \n" \
    "\n" \
    "If you'd like, I can also provide you with a list of some common topics that I can help with if you're looking for ideas. \n" \
    "\n" \
    "Just let me know if there's anything I can do to help make your day easier!",
  'generation_id' => '8a72b85c-ed29-4258-bd52-82b81be2353b',
  'token_count' => { 'prompt_tokens' => 64, 'response_tokens' => 82, 'total_tokens' => 146, 'billed_tokens' => 135 },
  'meta' => { 'api_version' => { 'version' => '1' },
              'billed_units' => { 'input_tokens' => 53, 'output_tokens' => 82 } },
  'tool_inputs' => nil }
```

## Index

- [TL;DR and Quick Start](#tldr-and-quick-start)
- [Index](#index)
- [Setup](#setup)
  - [Installing](#installing)
  - [Credentials](#credentials)
- [Usage](#usage)
  - [Client](#client)
    - [Custom Address](#custom-address)
  - [Methods](#methods)
    - [chat](#chat)
      - [Without Streaming Events](#without-streaming-events)
      - [Receiving Stream Events](#receiving-stream-events)
    - [generate](#generate)
    - [embed](#embed)
    - [rerank](#rerank)
    - [classify](#classify)
    - [detect_language](#detect_language)
    - [summarize](#summarize)
    - [tokenize](#tokenize)
    - [detokenize](#detokenize)
  - [Datasets](#datasets)
  - [Connectors](#connectors)
  - [Streaming and Server-Sent Events (SSE)](#streaming-and-server-sent-events-sse)
    - [Server-Sent Events (SSE) Hang](#server-sent-events-sse-hang)
  - [Back-and-Forth Conversations](#back-and-forth-conversations)
  - [New Functionalities and APIs](#new-functionalities-and-apis)
  - [Request Options](#request-options)
    - [Adapter](#adapter)
    - [Timeout](#timeout)
  - [Error Handling](#error-handling)
    - [Rescuing](#rescuing)
    - [For Short](#for-short)
    - [Errors](#errors)
- [Development](#development)
  - [Purpose](#purpose)
  - [Publish to RubyGems](#publish-to-rubygems)
  - [Updating the README](#updating-the-readme)
- [Resources and References](#resources-and-references)
- [Disclaimer](#disclaimer)

## Setup

### Installing

```sh
gem install cohere-ai -v 1.1.0
```

```sh
gem 'cohere-ai', '~> 1.1.0'
```

### Credentials

You can obtain your API key from the [Cohere AI Platform](https://dashboard.cohere.com).

## Usage

### Client

Ensure that you have an [API Key](#credentials) for authentication.

Create a new client:
```ruby
require 'cohere-ai'

client = Cohere.new(
  credentials: { api_key: ENV['COHERE_API_KEY'] },
  options: { server_sent_events: true }
)
```

#### Custom Address

You can use a custom address:

```ruby
require 'cohere-ai'

client = Cohere.new(
  credentials: {
    address: 'https://api.cohere.ai',
    api_key: ENV['COHERE_API_KEY']
  },
  options: { server_sent_events: true }
)
```

### Methods

#### chat

Documentation: https://docs.cohere.com/reference/chat

##### Without Streaming Events

```ruby
result = client.chat(
  { model: 'command', message: 'Hi!' }
)
```

Result:
```ruby
{ 'response_id' => '59aac3ec-74c8-4b65-a80c-d4c1916d2511',
  'text' =>
  "Hi there! I'm Coral, an AI-assistant chatbot ready to help you with whatever you need. Is there anything I can assist you with today? \n" \
    "\n" \
    "If you'd like, I can also provide you with a list of some common topics that I can help with if you're looking for ideas. \n" \
    "\n" \
    "Just let me know if there's anything I can do to help make your day easier!",
  'generation_id' => '8a72b85c-ed29-4258-bd52-82b81be2353b',
  'token_count' => { 'prompt_tokens' => 64, 'response_tokens' => 82, 'total_tokens' => 146, 'billed_tokens' => 135 },
  'meta' => { 'api_version' => { 'version' => '1' },
              'billed_units' => { 'input_tokens' => 53, 'output_tokens' => 82 } },
  'tool_inputs' => nil }
```

##### Receiving Stream Events

Ensure that you have enabled [Server-Sent Events](#streaming-and-server-sent-events-sse) before using blocks for streaming. You also need to add `stream: true` in your payload:

```ruby
client.chat(
  { model: 'command', stream: true, message: 'Hi!' }
) do |event, raw|
  puts event
end
```

Event:
```ruby
{ 'is_finished' => false,
  'event_type' => 'stream-start',
  'generation_id' => '0672df6f-736e-4536-8ad1-36bc808a114d' }
```

```ruby
{ 'is_finished' => false,
  'event_type' => 'text-generation',
  'text' => 'Hi' }
```

```ruby
{ 'is_finished' => true,
  'event_type' => 'stream-end',
  'response' => {
    'response_id' => '7dca6bf9-8a21-4f81-9e4e-2c5b8eb8c767',
    'text' => "Hi there! How can I help you today? I'm glad to assist you with any questions or tasks you have in mind, and I'm always happy to have a friendly conversation too. Go ahead and let me know how I can make your day better!",
    'generation_id' => '0672df6f-736e-4536-8ad1-36bc808a114d',
    'token_count' => {
      'prompt_tokens' => 64, 'response_tokens' => 52, 'total_tokens' => 116, 'billed_tokens' => 105
    },
    'tool_inputs' => nil
  },
  'finish_reason' => 'COMPLETE' }
```

You can get all the receive events at once as an array:
```ruby
result = client.chat(
  { model: 'command', stream: true, message: 'Hi!' }
)
```

Result:
```ruby
[{ 'is_finished' => false, 'event_type' => 'stream-start', 'generation_id' => '7f0349ef-a823-4779-801a-6e0a56c016a9' },
 { 'is_finished' => false, 'event_type' => 'text-generation', 'text' => 'Hi' },
 # ...
 { 'is_finished' => false, 'event_type' => 'text-generation', 'text' => '?' },
 { 'is_finished' => true,
   'event_type' => 'stream-end',
   'response' =>
   { 'response_id' => 'f4135de4-3631-4f7c-98be-deb50b7f986b',
     'text' =>
     "Hi there! How can I assist you today? I'm programmed to provide helpful, fact-based responses and engage in conversations on a wide range of topics. Feel free to ask me anything, and we'll see how I can help you out! \n" \
       "\n" \
       'Do you have any requests or questions that I can help with?',
     'generation_id' => '7f0349ef-a823-4779-801a-6e0a56c016a9',
     'token_count' => { 'prompt_tokens' => 64, 'response_tokens' => 65, 'total_tokens' => 129, 'billed_tokens' => 118 },
     'tool_inputs' => nil },
   'finish_reason' => 'COMPLETE' }]
```

You can mix both as well:
```ruby
result = client.chat(
  { model: 'command', stream: true, message: 'Hi!' }
) do |event, raw|
  puts event
end
```

#### generate

Documentation: https://docs.cohere.com/reference/generate

```ruby
client.generate(
  { stream: true,
    truncate: 'END',
    return_likelihoods: 'NONE',
    prompt: 'Please explain to me how LLMs work' }
) do |event, raw|
  puts event
end
```

```ruby
result = client.generate(
  { truncate: 'END',
    return_likelihoods: 'NONE',
    prompt: 'Please explain to me how LLMs work' }
)
```

Result:
```ruby
{ 'id' => '9caef8a6-07fb-465c-a003-5fd7c7bbccbd',
  'generations' =>
  [{ 'id' => '11b043ab-6d9c-415f-90b0-6a802a3807f7',
     'text' =>
     " LLMs, or Large Language Models, are a type of neural network architecture designed to understand and generate human-like language. They are trained by feeding them massive amounts of text data and adjusting their internal parameters to predict the next word in a sequence accurately, given the words that came before it. This process is known as language modeling.\n" \
       "\n" \
       "There are two main types of LLMs: autoregressive and generative models. Autoregressive models, such as the long short-term memory (LSTM) network and the gated recurrent unit (GRU), predict the next word in a sequence by relying solely on the past context. On the other hand, generative models, such as the variational autoencoder (VAE) and the generative adversarial network (GAN), can generate new data similar to the training data.\n" \
       "\n" \
       "One of the key features of LLMs is their ability to capture long-range dependencies in text, which means they can understand and generate sentences that rely on information from words far away in the sequence. This is particularly useful for tasks like language translation, summarization, and question-answering, where understanding the context is crucial.\n" \
       "\n" \
       "To improve the accuracy and performance of LLMs, researchers and engineers often use techniques like pre-training and transfer learning. During pre-training, the LLM is trained on a large and diverse dataset, such as books, articles, and websites, to learn general language patterns and relationships. Once the LLM is pre-trained, it can be fine-tuned on specific tasks with smaller datasets related to the task, such as medical texts or legal documents. This way, the LLM can leverage its knowledge of language understanding while adapting to the specific domain or task at hand.\n" \
       "\n" \
       "Overall, LLMs have had a significant impact on the field of natural language processing, allowing researchers to build more sophisticated models that can understand and generate language in ways that were previously impossible. However, it is important to note that LLMs are sophisticated tools that require large amounts of data, computational power, and careful tuning to achieve their full potential. Moreover, they can also inherit the biases present in the training data, highlighting the importance of data diversity and responsible LLM development and usage. \n" \
       "\n" \
       'Would you like me to go into more detail about any specific aspect of LLM functionality? ',
     'finish_reason' => 'COMPLETE' }],
  'prompt' => 'Please explain to me how LLMs work',
  'meta' => { 'api_version' => { 'version' => '1' },
              'billed_units' => { 'input_tokens' => 8, 'output_tokens' => 476 } } }
```

#### embed

Documentation: https://docs.cohere.com/reference/embed

```ruby
result = client.embed(
  { texts: ['hello', 'goodbye'],
    model: 'embed-english-v3.0',
    input_type: 'classification' }
)
```

Result:
```ruby
{ 'id' => '5d77f09a-3518-44d3-bac3-f868b3e036bc',
  'texts' => ['hello', 'goodbye'],
  'embeddings' =>
  [[0.016296387,
    -0.008354187,
    # ...
    -0.01802063,
    0.009765625],
   [0.04663086,
    -0.023239136,
    # ...
    0.0023212433,
    0.0052719116]],
  'meta' =>
  { 'api_version' => { 'version' => '1' }, 'billed_units' => { 'input_tokens' => 2 } },
  'response_type' => 'embeddings_floats' }
```

#### rerank

Documentation: https://docs.cohere.com/reference/rerank

```ruby
result = client.rerank(
  { return_documents: false,
    max_chunks_per_doc: 10,
    model: 'rerank-english-v2.0',
    query: 'What is the capital of the United States?',
    documents: [
      'Carson City is the capital city of the American state of Nevada.',
      'The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.',
      'Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.',
      'Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.'
    ] }
)
```

Result:
```ruby
{ 'id' => 'ba2100fe-69d2-41d6-8680-810948af872d',
  'results' =>
  [{ 'index' => 2, 'relevance_score' => 0.98005307 },
   { 'index' => 3, 'relevance_score' => 0.27904198 },
   { 'index' => 0, 'relevance_score' => 0.10194652 },
   { 'index' => 1, 'relevance_score' => 0.0721122 }],
  'meta' =>
  { 'api_version' => { 'version' => '1' }, 'billed_units' => { 'search_units' => 1 } } }
```

#### classify

Documentation: https://docs.cohere.com/reference/classify

```ruby
result = client.classify(
  {
    truncate: 'END',
    inputs: [
      'Confirm your email address',
      'hey i need u to send some $'
    ],
    examples: [
      {
        text: "Dermatologists don't like her!",
        label: 'Spam'
      },
      {
        text: 'Hello, open to this?',
        label: 'Spam'
      },
      {
        text: 'I need help please wire me $1000 right now',
        label: 'Spam'
      },
      {
        text: 'Nice to know you ;)',
        label: 'Spam'
      },
      {
        text: 'Please help me?',
        label: 'Spam'
      },
      {
        text: 'Your parcel will be delivered today',
        label: 'Not spam'
      },
      {
        text: 'Review changes to our Terms and Conditions',
        label: 'Not spam'
      },
      {
        text: 'Weekly sync notes',
        label: 'Not spam'
      },
      {
        text: 'Re: Follow up from today’s meeting',
        label: 'Not spam'
      },
      {
        text: 'Pre-read for tomorrow',
        label: 'Not spam'
      }
    ]
  }
)
```

Result:
```ruby
{ 'id' => '1d0c322e-9ab7-4a80-b601-720e7592dde5',
  'classifications' =>
  [{ 'classification_type' => 'single-label',
     'confidence' => 0.8082329,
     'confidences' => [0.8082329],
     'id' => '44b2e358-83f1-4b9b-a6f1-e8d9c823f0ba',
     'input' => 'Confirm your email address',
     'labels' =>
     { 'Not spam' => { 'confidence' => 0.8082329 },
       'Spam' => { 'confidence' => 0.19176713 } },
     'prediction' => 'Not spam',
     'predictions' => ['Not spam'] },
   { 'classification_type' => 'single-label',
     'confidence' => 0.9893421,
     'confidences' => [0.9893421],
     'id' => '53ad6d6e-872f-4185-b3d6-6d03fe55f4c4',
     'input' => 'hey i need u to send some $',
     'labels' =>
     { 'Not spam' => { 'confidence' => 0.01065793 },
       'Spam' => { 'confidence' => 0.9893421 } },
     'prediction' => 'Spam',
     'predictions' => ['Spam'] }],
  'meta' =>
  { 'api_version' => { 'version' => '1' }, 'billed_units' => { 'classifications' => 2 } } }
```

#### detect_language

Documentation: https://docs.cohere.com/reference/detect-language

```ruby
result = client.detect_language(
  { texts: ['Hello world', "'Здравствуй, Мир'"] }
)
```

Result:
```ruby
{ 'id' => '82b111d4-ba6d-4f60-9536-883b42f3d86d',
  'results' =>
  [{ 'language_code' => 'en', 'language_name' => 'English' },
   { 'language_code' => 'ru', 'language_name' => 'Russian' }],
  'meta' => { 'api_version' => { 'version' => '1' } } }
```

#### summarize

Documentation: https://docs.cohere.com/reference/summarize

```ruby
result = client.summarize(
  { text: "Ice cream is a sweetened frozen food typically eaten as a snack or dessert. It may be made from milk or cream and is flavoured with a sweetener, either sugar or an alternative, and a spice, such as cocoa or vanilla, or with fruit such as strawberries or peaches. It can also be made by whisking a flavored cream base and liquid nitrogen together. Food coloring is sometimes added, in addition to stabilizers. The mixture is cooled below the freezing point of water and stirred to incorporate air spaces and to prevent detectable ice crystals from forming. The result is a smooth, semi-solid foam that is solid at very low temperatures (below 2 °C or 35 °F). It becomes more malleable as its temperature increases.\n\nThe meaning of the name \"ice cream\" varies from one country to another. In some countries, such as the United States, \"ice cream\" applies only to a specific variety, and most governments regulate the commercial use of the various terms according to the relative quantities of the main ingredients, notably the amount of cream. Products that do not meet the criteria to be called ice cream are sometimes labelled \"frozen dairy dessert\" instead. In other countries, such as Italy and Argentina, one word is used fo\r all variants. Analogues made from dairy alternatives, such as goat's or sheep's milk, or milk substitutes (e.g., soy, cashew, coconut, almond milk or tofu), are available for those who are lactose intolerant, allergic to dairy protein or vegan." }
)
```

Result:
```ruby
{ 'id' => '5189bed8-2d12-47d8-8ae7-cf60788cf507',
  'summary' =>
  'Ice cream is a popular frozen dessert made from milk or cream, sweeteners, and spices like vanilla or cocoa, or fruit like strawberries. It is cooled below the freezing point of water and stirred to incorporate air spaces, resulting in a smooth semi-solid foam. While the name applies to just one variety in some countries, like the US, it refers to all variants in others, like Italy. Additionally, there are dairy alternatives available for those who are lactose intolerant or vegan. Ice cream is regulated by governments based on the quantities of its main ingredients.',
  'meta' =>
  { 'api_version' => { 'version' => '1' },
    'billed_units' => { 'input_tokens' => 321, 'output_tokens' => 111 } } }
```

#### tokenize

Documentation: https://docs.cohere.com/reference/tokenize

```ruby
result = client.tokenize(
  { text: 'tokenize me! :D', model: 'command' }
)
```

Result:
```ruby
{ 'tokens' => [10_002, 2261, 2012, 8, 2792, 43],
  'token_strings' => ['token', 'ize', ' me', '!', ' :', 'D'],
  'meta' => { 'api_version' => { 'version' => '1' } } }
```

#### detokenize

Documentation: https://docs.cohere.com/reference/detokenize

```ruby
result = client.detokenize(
  { tokens: [10_104, 12_221, 1315, 34, 1420, 69], model: 'command' }
)
```

Result:
```ruby
{ 'text' => ' Anton Mun🟣;🥭^', 'meta' => { 'api_version' => { 'version' => '1' } } }
```

### Datasets

Documentation: https://docs.cohere.com/reference/create-dataset

```ruby
result = client.request(
  'v1/dataset',
  request_method: 'GET', server_sent_events: false
)
```

```ruby
result = client.request(
  'v1/dataset',
  { name: 'prompt-completion-dataset',
    data: File.read('./prompt-completion.jsonl'),
    dataset_type: 'prompt-completion-finetune-input' },
  request_method: 'POST', server_sent_events: false
)
```

### Connectors

Documentation: https://docs.cohere.com/reference/list-connectors

```ruby
result = client.request(
  'v1/connectors',
  request_method: 'GET', server_sent_events: false
)
```

```ruby
result = client.request(
  'v1/connectors',
  { name: 'test-connector',
    url: 'https://example.com/search',
    description: 'A test connector' },
  request_method: 'POST', server_sent_events: false
)
```

### Streaming and Server-Sent Events (SSE)

[Server-Sent Events (SSE)](https://en.wikipedia.org/wiki/Server-sent_events) is a technology that allows certain endpoints to offer streaming capabilities, such as creating the impression that "the model is typing along with you," rather than delivering the entire answer all at once.

You can set up the client to use Server-Sent Events (SSE) for all supported endpoints:
```ruby
client = Cohere.new(
  credentials: { api_key: ENV['COHERE_API_KEY'] },
  options: { server_sent_events: true }
)
```

Or, you can decide on a request basis:
```ruby
client.chat(
  { model: 'command', stream: true, message: 'Hi!' }
  server_sent_events: true
) do |event, raw|
  puts event
end
```

With Server-Sent Events (SSE) enabled, you can use a block to receive partial results via events. This feature is particularly useful for methods that offer streaming capabilities, such as `chat`: [Receiving Stream Events](#receiving-stream-events)

#### Server-Sent Events (SSE) Hang

Method calls will _hang_ until the server-sent events finish, so even without providing a block, you can obtain the final results of the received events: [Receiving Stream Events](#receiving-stream-events)


### Back-and-Forth Conversations

To maintain a back-and-forth conversation, you need to append the received responses and build a history for your requests:

```rb
result = client.chat(
  { model: 'command',
    chat_history: [
      { role: 'USER', message: 'Hi, my name is Purple.' },
      { role: 'CHATBOT', message: "Hi Purple! It's nice to meet you." }
    ],
    message: "What's my name?" }
)
```

Result:
```ruby
{ 'response_id' => 'a2009c24-fee3-465e-9945-a85edcdcb3cf',
  'text' =>
  "Your name is Purple. Isn't it an enchanting name? \n" \
    "\n" \
    "Would you like me to help you with anything else? If you have any questions or need assistance with a particular task, feel free to let me know. I'm here to help!",
  'generation_id' => 'e1630c10-f773-4ced-8760-19199004653a',
  'token_count' => { 'prompt_tokens' => 91, 'response_tokens' => 51, 'total_tokens' => 142, 'billed_tokens' => 124 },
  'meta' => { 'api_version' => { 'version' => '1' },
              'billed_units' => { 'input_tokens' => 73, 'output_tokens' => 51 } },
  'tool_inputs' => nil }

```

### New Functionalities and APIs

Cohere may launch a new endpoint that we haven't covered in the Gem yet. If that's the case, you may still be able to use it through the `request` method. For example, `chat` is just a wrapper for `v1/chat`, which you can call directly like this:

```ruby
result = client.request(
  'v1/chat',
  { model: 'command', message: 'Hi!' },
  request_method: 'POST', server_sent_events: true
)
```

### Request Options

#### Adapter

The gem uses [Faraday](https://github.com/lostisland/faraday) with the [Typhoeus](https://github.com/typhoeus/typhoeus) adapter by default.

You can use a different adapter if you want:

```ruby
require 'faraday/net_http'

client = Cohere.new(
  credentials: { api_key: ENV['COHERE_API_KEY'] },
  options: { connection: { adapter: :net_http } }
)
```

#### Timeout

You can set the maximum number of seconds to wait for the request to complete with the `timeout` option:

```ruby
client = Cohere.new(
  credentials: { api_key: ENV['COHERE_API_KEY'] },
  options: { connection: { request: { timeout: 5 } } }
)
```

You can also have more fine-grained control over [Faraday's Request Options](https://lostisland.github.io/faraday/#/customization/request-options?id=request-options) if you prefer:

```ruby
client = Cohere.new(
  credentials: { api_key: ENV['COHERE_API_KEY'] },
  options: {
    connection: {
      request: {
        timeout: 5,
        open_timeout: 5,
        read_timeout: 5,
        write_timeout: 5
      }
    }
  }
)
```

### Error Handling

#### Rescuing

```ruby
require 'cohere-ai'

begin
  client.chat(
    { model: 'command', message: 'Hi!' }
  )
rescue Cohere::Errors::CohereError => error
  puts error.class # Cohere::Errors::RequestError
  puts error.message # 'the server responded with status 500'

  puts error.payload
  # { model: 'command',
  #   message: 'Hi!'
  #   ...
  # }

  puts error.request
  # #<Faraday::ServerError response={:status=>500, :headers...
end
```

#### For Short

```ruby
require 'cohere-ai/errors'

begin
  client.chat(
    { model: 'command',
      messages: [{ role: 'user', content: 'hi!' }] }
  )
rescue CohereError => error
  puts error.class # Cohere::Errors::RequestError
end
```

#### Errors

```ruby
CohereError

MissingAPIKeyError
BlockWithoutServerSentEventsError
IncompleteJSONReceivedError

RequestError
```

## Development

```bash
bundle
rubocop -A

bundle exec ruby spec/tasks/run-client.rb
```

### Purpose

This Gem is designed to provide low-level access to Cohere, enabling people to build abstractions on top of it. If you are interested in more high-level abstractions or more user-friendly tools, you may want to consider [Nano Bots](https://github.com/icebaker/ruby-nano-bots) 💎 🤖.

### Publish to RubyGems

```bash
gem build cohere-ai.gemspec

gem signin

gem push cohere-ai-1.1.0.gem
```

### Updating the README

Install [Babashka](https://babashka.org):

```sh
curl -s https://raw.githubusercontent.com/babashka/babashka/master/install | sudo bash
```

Update the `template.md` file and then:

```sh
bb tasks/generate-readme.clj
```

Trick for automatically updating the `README.md` when `template.md` changes:

```sh
sudo pacman -S inotify-tools # Arch / Manjaro
sudo apt-get install inotify-tools # Debian / Ubuntu / Raspberry Pi OS
sudo dnf install inotify-tools # Fedora / CentOS / RHEL

while inotifywait -e modify template.md; do bb tasks/generate-readme.clj; done
```

Trick for Markdown Live Preview:
```sh
pip install -U markdown_live_preview

mlp README.md -p 8076
```

## Resources and References

These resources and references may be useful throughout your learning process.

- [Cohere AI Official Website](https://cohere.com)
- [Cohere AI Documentation](https://docs.cohere.com/docs)
- [Cohere AI API Reference](https://docs.cohere.com/reference/about)

## Disclaimer

This is not an official Cohere project, nor is it affiliated with Cohere in any way.

This software is distributed under the [MIT License](https://github.com/gbaptista/cohere-ai/blob/main/LICENSE). This license includes a disclaimer of warranty. Moreover, the authors assume no responsibility for any damage or costs that may result from using this project. Use the Cohere AI Ruby Gem at your own risk.
