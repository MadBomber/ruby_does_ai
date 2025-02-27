# Maritaca AI

A Ruby gem for interacting with [MariTalk](https://chat.maritaca.ai) from [Maritaca AI](https://www.maritaca.ai).

![The image features a minimalist logo combining a red ruby and a green Brazilian maritaca bird. The left side shows a flat, smooth red ruby, while the right side transitions to a vibrant green maritaca bird, both depicted in a simple, stylized manner. The background is a subtle gradient, enhancing the logo's modern and clean design..](https://raw.githubusercontent.com/gbaptista/assets/main/maritaca-ai/maritaca-ai-canvas.png)

> _This Gem is designed to provide low-level access to MariTalk, enabling people to build abstractions on top of it. If you are interested in more high-level abstractions or more user-friendly tools, you may want to consider [Nano Bots](https://github.com/icebaker/ruby-nano-bots) 💎 🤖._

## TL;DR and Quick Start

```ruby
gem 'maritaca-ai', '~> 1.2.0'
```

```ruby
require 'maritaca-ai'

client = Maritaca.new(
  credentials: { api_key: ENV['MARITACA_API_KEY'] },
  options: { server_sent_events: true }
)

result = client.chat_inference(
  { model: 'sabia-2-medium',
    chat_mode: true,
    messages: [ { role: 'user', content: 'Oi!' } ] }
)
```

Result:
```ruby
{ 'answer' => ' Oi! Como posso ajudar você hoje?',
  'usage' => {
    'completion_tokens' => 15,
    'prompt_tokens' => 3,
    'total_tokens' => 18
  },
  'model' => 'sabia-2-medium' }
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
    - [chat_inference](#chat_inference)
      - [Without Streaming Events](#without-streaming-events)
        - [Chat](#chat)
        - [Back-and-Forth Conversations](#back-and-forth-conversations)
        - [Without Chat](#without-chat)
      - [Receiving Stream Events](#receiving-stream-events)
  - [Streaming and Server-Sent Events (SSE)](#streaming-and-server-sent-events-sse)
    - [Server-Sent Events (SSE) Hang](#server-sent-events-sse-hang)
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
gem install maritaca-ai -v 1.2.0
```

```sh
gem 'maritaca-ai', '~> 1.2.0'
```

### Credentials

You can obtain your API key at [MariTalk](https://chat.maritaca.ai).

Enclose credentials in single quotes when using environment variables to prevent issues with the `$` character in the API key:

```sh
MARITACA_API_KEY='123...$a12...'
```

## Usage

### Client

Ensure that you have an [API Key](#credentials) for authentication.

Create a new client:
```ruby
require 'maritaca-ai'

client = Maritaca.new(
  credentials: { api_key: ENV['MARITACA_API_KEY'] },
  options: { server_sent_events: true }
)
```

#### Custom Address

You can use a custom address:

```ruby
require 'maritaca-ai'

client = Maritaca.new(
  credentials: {
    address: 'https://chat.maritaca.ai',
    api_key: ENV['MARITACA_API_KEY']
  }
)
```

### Methods

#### chat_inference

##### Without Streaming Events

###### Chat

```ruby
result = client.chat_inference(
  { model: 'sabia-2-medium',
    chat_mode: true,
    messages: [ { role: 'user', content: 'Oi!' } ] }
)
```

Result:
```ruby
{ 'answer' => ' Oi! Como posso ajudar você hoje?',
  'usage' => {
    'completion_tokens' => 15,
    'prompt_tokens' => 3,
    'total_tokens' => 18
  },
  'model' => 'sabia-2-medium' }
```

###### Back-and-Forth Conversations

To maintain a back-and-forth conversation, you need to append the received responses and build a history for your requests:

```rb
result = client.chat_inference(
  { model: 'sabia-2-medium',
    chat_mode: true,
    messages: [
      { role: 'user', content: 'Oi, meu nome é Tamanduá.' },
      { role: 'assistant', content: 'Oi Tamanduá, como posso ajudá-lo hoje?' },
      { role: 'user', content: 'Qual é o meu nome?' }
    ] }
)
```

Result:
```ruby
{ 'answer' => ' Seu nome é Tamanduá. É um prazer conhecê-lo! Como posso ajudá-lo hoje?',
  'usage' => {
    'completion_tokens' => 35,
    'prompt_tokens' => 39,
    'total_tokens' => 74
  },
  'model' => 'sabia-2-medium' }
```

###### Without Chat

You can prompt the model without using chat mode:

```ruby
result = client.chat_inference(
  { model: 'sabia-2-medium',
    chat_mode: false,
    messages: "Minha terra tem palmeiras,\nOnde canta o Sabiá;\n",
    stopping_tokens: ['.'] }
)
```

Result:
```ruby
{ 'answer' =>
    "As aves, que aqui gorjeiam,\n" \
    'Não gorjeiam como lá.',
  'usage' => {
    'completion_tokens' => 21,
    'prompt_tokens' => 21,
    'total_tokens' => 42
  },
  'model' => 'sabia-2-medium' }
```

##### Receiving Stream Events

Ensure that you have enabled [Server-Sent Events](#streaming-and-server-sent-events-sse) before using blocks for streaming. You also need to add `stream: true` in your payload:

```ruby
client.chat_inference(
  { model: 'sabia-2-medium',
    stream: true,
    chat_mode: true,
    messages: [ { role: 'user', content: 'Oi!' } ] }
) do |event, parsed, raw|
  puts event
end
```

Event:
```ruby
{ 'text' => ' Oi! Com' }
```

You can get all the receive events at once as an array:

```ruby
result = client.chat_inference(
  { model: 'sabia-2-medium',
    stream: true,
    chat_mode: true,
    messages: [ { role: 'user', content: 'Oi!' } ] }
)
```

Result:
```ruby
[{ 'text' => ' Oi! Com' },
 { 'text' => 'o posso a' },
 { 'text' => 'judar você' },
 { 'text' => ' hoje?' },
 { 'completion_tokens' => 15,
   'prompt_tokens' => 74,
   'total_tokens' => 89,
   'model' => 'sabia-2-medium' }]
```

You can mix both as well:
```ruby
result = client.chat_inference(
  { model: 'sabia-2-medium',
    stream: true,
    chat_mode: true,
    messages: [ { role: 'user', content: 'Oi!' } ] }
) do |event, parsed, raw|
  puts event
end
```

### Streaming and Server-Sent Events (SSE)

[Server-Sent Events (SSE)](https://en.wikipedia.org/wiki/Server-sent_events) is a technology that allows certain endpoints to offer streaming capabilities, such as creating the impression that "the model is typing along with you," rather than delivering the entire answer all at once.

You can set up the client to use Server-Sent Events (SSE) for all supported endpoints:
```ruby
client = Maritaca.new(
  credentials: { api_key: ENV['MARITACA_API_KEY'] },
  options: { server_sent_events: true }
)
```

Or, you can decide on a request basis:
```ruby
client.chat_inference(
  { model: 'sabia-2-medium',
    stream: true,
    chat_mode: true,
    messages: [ { role: 'user', content: 'Oi!' } ] },
  server_sent_events: true
) do |event, parsed, raw|
  puts event
end
```

With Server-Sent Events (SSE) enabled, you can use a block to receive partial results via events. This feature is particularly useful for methods that offer streaming capabilities, such as `chat_inference`: [Receiving Stream Events](#receiving-stream-events)

#### Server-Sent Events (SSE) Hang

Method calls will _hang_ until the server-sent events finish, so even without providing a block, you can obtain the final results of the received events: [Receiving Stream Events](#receiving-stream-events)

### New Functionalities and APIs

Maritaca may launch a new endpoint that we haven't covered in the Gem yet. If that's the case, you may still be able to use it through the `request` method. For example, `chat_inference` is just a wrapper for `api/chat/inference`, which you can call directly like this:

```ruby
result = client.request(
  'api/chat/inference',
  { model: 'sabia-2-medium',
    chat_mode: true,
    messages: [{ role: 'user', content: 'Oi!' }] },
  request_method: 'POST'
)
```

### Request Options

#### Adapter

The gem uses [Faraday](https://github.com/lostisland/faraday) with the [Typhoeus](https://github.com/typhoeus/typhoeus) adapter by default.

You can use a different adapter if you want:

```ruby
require 'faraday/net_http'

client = Maritaca.new(
  credentials: { api_key: ENV['MARITACA_API_KEY'] },
  options: { connection: { adapter: :net_http } }
)
```

#### Timeout

You can set the maximum number of seconds to wait for the request to complete with the `timeout` option:

```ruby
client = Maritaca.new(
  credentials: { api_key: ENV['MARITACA_API_KEY'] },
  options: { connection: { request: { timeout: 5 } } }
)
```

You can also have more fine-grained control over [Faraday's Request Options](https://lostisland.github.io/faraday/#/customization/request-options?id=request-options) if you prefer:

```ruby
client = Maritaca.new(
  credentials: { api_key: ENV['MARITACA_API_KEY'] },
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
require 'maritaca-ai'

begin
  client.chat_inference(
    { model: 'sabia-2-medium',
      chat_mode: true,
      messages: [ { role: 'user', content: 'Oi!' } ] }
  )
rescue Maritaca::Errors::MaritacaError => error
  puts error.class # Maritaca::Errors::RequestError
  puts error.message # 'the server responded with status 500'

  puts error.payload
  # { model: 'sabia-2-medium',
  #   chat_mode: true,
  #   ...
  # }

  puts error.request
  # #<Faraday::ServerError response={:status=>500, :headers...
end
```

#### For Short

```ruby
require 'maritaca-ai/errors'

begin
  client.chat_inference(
    { model: 'sabia-2-medium',
      chat_mode: true,
      messages: [ { role: 'user', content: 'Oi!' } ] }
  )
rescue MaritacaError => error
  puts error.class # Maritaca::Errors::RequestError
end
```

#### Errors

```ruby
MaritacaError

MissingAPIKeyError

RequestError
```

## Development

```bash
bundle
rubocop -A

bundle exec ruby spec/tasks/run-client.rb
```

### Purpose

This Gem is designed to provide low-level access to MariTalk, enabling people to build abstractions on top of it. If you are interested in more high-level abstractions or more user-friendly tools, you may want to consider [Nano Bots](https://github.com/icebaker/ruby-nano-bots) 💎 🤖.

### Publish to RubyGems

```bash
gem build maritaca-ai.gemspec

gem signin

gem push maritaca-ai-1.2.0.gem
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

- [Maritaca AI Official Website](https://www.maritaca.ai)
- [MariTalk Documentation](https://maritaca-ai.github.io/maritalk-api/maritalk.html)
- [Maritaca AI API Documentation](https://chat.maritaca.ai/docs)

## Disclaimer

This is not an official Maritaca AI project, nor is it affiliated with Maritaca AI in any way.

This software is distributed under the [MIT License](https://github.com/gbaptista/maritaca-ai/blob/main/LICENSE). This license includes a disclaimer of warranty. Moreover, the authors assume no responsibility for any damage or costs that may result from using this project. Use the Maritaca AI Ruby Gem at your own risk.
