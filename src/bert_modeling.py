from torch import nn


class BottleneckAdapter(nn.Module):
    def __init__(self, config):
        super(BottleneckAdapter, self).__init__()
        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        self.activation = config.adapter_act
        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        self.init_weights(config)

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return up_projected

    def init_weights(self, config):
        self.down_project.weight.data.normal_(
            mean=0.0, std=config.adapter_initializer_range
        )
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(
            mean=0.0, std=config.adapter_initializer_range
        )
        self.up_project.bias.data.zero_()


class BertOutput(nn.Module):
    def __init__(self, output, adapter=None):
        super(BertOutput, self).__init__()
        self.dense = output.dense
        self.dropout = output.dropout
        self.LayerNorm = output.LayerNorm
        self.use_adapter = False

        if adapter is not None:
            self.adapter = adapter
            self.use_adapter = True

    def forward(self, hidden_states, residual_input):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.use_adapter:
            hidden_states = self.adapter(hidden_states) + hidden_states

        hidden_states = self.LayerNorm(hidden_states + residual_input)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, layer, adapter=None):
        super(BertLayer, self).__init__()
        self.attention = layer.attention
        self.intermediate = layer.intermediate
        self.output = BertOutput(layer.output, adapter)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(
            hidden_states=hidden_states, attention_mask=attention_mask
        )[0]

        intermediate_output = self.intermediate(attention_output)
        hidden_states = self.output(intermediate_output, attention_output)

        return hidden_states


class BertEncoder(nn.Module):
    def __init__(self, encoder, adapter=None):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList(
            [BertLayer(layer, adapter) for layer in encoder.layer]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class BertModel(nn.Module):
    def __init__(self, bert, adapter=None):
        super(BertModel, self).__init__()
        self.embeddings = bert.embeddings
        self.encoder = BertEncoder(bert.encoder, adapter)
        self.pooler = bert.pooler
        # self.prompt_tuning = bert.prompt_tuning

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

        extended_attention_mask = (
            attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=embedding_output.dtype)
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output, extended_attention_mask)

        sequence_output = encoded_layers[-1]

        pooled_output = self.pooler(sequence_output)

        # prompt_output = self.prompt_tuning(pooled_output)

        return encoded_layers, pooled_output


class ClassificationHead(nn.Module):
    def __init__(self, dropout, classifier):
        super(ClassificationHead, self).__init__()
        self.dropout = dropout
        self.classifier = classifier

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits


class AdapterBertForSequenceClassification(nn.Module):
    def __init__(self, bert_model, classifier_head):
        super(AdapterBertForSequenceClassification, self).__init__()
        self.bert_model = bert_model
        self.classifier_head = classifier_head

    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.classifier_head(pooled_output[1])
        return logits
